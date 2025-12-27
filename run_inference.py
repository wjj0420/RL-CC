import os
import warnings

# --- 1. 配置屏蔽警告 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

import numpy as np
import sysv_ipc
import time
import math
import collections
from agent import Agent

# --- 2. 核心参数配置 ---
STATE_DIM_SINGLE = 7      # 单帧特征维度
REC_DIM = 10              # 回溯帧数
STATE_DIM_TOTAL = 70      # 7 * 10
ACTION_DIM = 1
H1_SHAPE = 256
H2_SHAPE = 256

MEM_KEY_READ = 123456     # 读状态 Key
MEM_KEY_WRITE = 12345     # 写动作 Key
MODEL_PATH = "./PCC_models/"  # 模型路径

# --- 3. 定义特征处理类 (StateProcessor) ---
class StateProcessor:
    def __init__(self):
        self.max_bw = 1.0  # 初始设为1，避免除零
        
    def process_state(self, raw_15_params):
        """
        将 C++ 发来的 15 个原始参数转换为模型需要的 7 个归一化特征
        """
        # 解包参数 (基于 envwrapper.py 和 orca-server-mahimahi.cc 的顺序)
        # [0]delay, [1]thr, [2]samples, [3]delta_t, [4]target, [5]cwnd
        # [6]pacing, [7]lost_rate(Bps), [8]srtt, ... [14]min_rtt
        
        thr = raw_15_params[1]
        samples = raw_15_params[2]
        delta_t = raw_15_params[3]
        cwnd = raw_15_params[5]
        pacing = raw_15_params[6]
        lost_rate = raw_15_params[7]
        srtt = raw_15_params[8]
        min_rtt = raw_15_params[14]

        # 动态更新最大带宽 (Online Normalization)
        if thr > self.max_bw:
            self.max_bw = thr

        # 构造 7 维特征
        state = np.zeros(7)

        # 1. 归一化吞吐量
        state[0] = thr / self.max_bw if self.max_bw > 0 else 0
        
        # 2. 归一化 Pacing Rate (截断为 10)
        val = pacing / self.max_bw if self.max_bw > 0 else 0
        state[1] = min(val, 10.0)

        # 3. 归一化丢包率 (系数 5)
        state[2] = (5 * lost_rate) / self.max_bw if self.max_bw > 0 else 0

        # 4. 采样效率 (Samples / CWND)
        state[3] = samples / cwnd if cwnd > 0 else 0

        # 5. 采样间隔
        state[4] = delta_t

        # 6. 延迟比 (MinRTT / SRTT)
        state[5] = min_rtt / srtt if srtt > 0 else 1.0

        # 7. 延迟惩罚度量
        delay_margin = 1.25
        if min_rtt * delay_margin < srtt:
             state[6] = (min_rtt * delay_margin) / srtt
        else:
             state[6] = 1.0

        return state

    def process_action(self, raw_action):
        """
        将模型输出的 float (例如 0.5) 转换为 C++ 需要的 int (例如 200)
        公式: 4^action * 100
        """
        multiplier = math.pow(4, raw_action)
        return int(multiplier * 100)

# --- 4. 主逻辑 ---
def main():
    # === 初始化资源 ===
    print("正在连接共享内存...")
    while True:
        try:
            shm_r = sysv_ipc.SharedMemory(MEM_KEY_READ)
            # 创建或连接写内存（如果不存在则创建）
            try:
                shm_w = sysv_ipc.SharedMemory(MEM_KEY_WRITE)
            except sysv_ipc.ExistentialError:
                # 如果不存在则创建
                shm_w = sysv_ipc.SharedMemory(MEM_KEY_WRITE, flags=sysv_ipc.IPC_CREAT, mode=0o666, size=512)
            print("共享内存连接成功!")
            break
        except sysv_ipc.ExistentialError:
            time.sleep(1)
            print(".", end="", flush=True)

    # === 初始化模型 ===
    tf_config = tf.ConfigProto()


    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    agent = Agent(s_dim=STATE_DIM_TOTAL, a_dim=ACTION_DIM, batch_size=1, 
                  h1_shape=H1_SHAPE, h2_shape=H2_SHAPE)
    
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(MODEL_PATH)
    if checkpoint:
        saver.restore(sess, checkpoint)
        print(f"模型已加载: {checkpoint}")
    else:
        print("错误: 未找到模型文件！")
        return

    agent.assign_sess(sess)

    # === 初始化处理器与缓冲区 ===
    processor = StateProcessor()
    
    # 初始化状态缓冲区 (全0填充)
    # 这里的 maxlen=10 会自动挤掉旧数据，保留最新的
    state_buffer = collections.deque(maxlen=REC_DIM)
    for _ in range(REC_DIM):
        state_buffer.append(np.zeros(STATE_DIM_SINGLE))

    last_id = -1
    print("\n------- 推理循环启动 -------")

    # === 5. 详细的集成主循环 ===
    while True:
        try:
            # --- 步骤 A: 读取原始字节流 ---
            try:
                raw_bytes = shm_r.read()
            except Exception:
                break # 内存可能被删除了
            
            # 解码字符串 (去除末尾空字符)
            raw_str = raw_bytes.decode('utf-8', errors='ignore').split('\0')[0].strip()
            
            if not raw_str:
                time.sleep(0.001)
                continue

            # --- 步骤 B: 解析为数字列表 ---
            try:
                data = list(map(float, raw_str.split()))
            except ValueError:
                time.sleep(0.001)
                continue

            # 检查长度 (ID + 15个参数 = 16)
            if len(data) < 16:
                time.sleep(0.001)
                continue

            curr_id = int(data[0])

            # --- 步骤 C: 检查是否有新数据 ---
            if curr_id == last_id:
                # ID 没变，说明 C++ 还没产生新状态，休息一下
                time.sleep(0.001)
                continue

            # ==========================================
            #      核心集成点：数据处理与推理
            # ==========================================

            # 1. 提取原始参数 (去掉开头的 ID)
            raw_15_params = np.array(data[1:]) 

            # 2. 【调用 processor】生成 7 维特征
            current_7_features = processor.process_state(raw_15_params)

            # 3. 更新缓冲区 (自动把这一帧加到队尾，最早的一帧移出)
            state_buffer.append(current_7_features)

            # 4. 拼接成模型需要的 70 维向量 (Flatten)
            # [Frame_t-9, Frame_t-8, ..., Frame_t]
            combined_input = np.concatenate(state_buffer)

            # 5. 模型推理
            # 注意: get_action 期望输入是 batch 形式，所以加个 [] 变成 [1, 70]
            # use_noise=False 表示纯推理，不加随机探索噪声
            raw_action = agent.get_action([combined_input], use_noise=False)[0][0]

            # 6. 【调用 processor】处理动作
            final_action_int = processor.process_action(raw_action)

            # ==========================================

            # --- 步骤 D: 写回共享内存 ---
            # 格式: "ID Action\0"
            msg = f"{curr_id} {final_action_int}\0"
            shm_w.write(msg)
            
            # --- 调试打印 (可选) ---
            # print(f"Step: {curr_id} | In: {current_7_features[:2]}... | Out: {raw_action:.4f} -> {final_action_int}")

            # 更新 ID
            last_id = curr_id

        except KeyboardInterrupt:
            print("\n停止。")
            break
        except Exception as e:
            print(f"Loop Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()