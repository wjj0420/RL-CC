import sysv_ipc
import time
import random
import struct

# 必须与 run_inference.py 一致
MEM_KEY_READ = 123456  # 写入状态供模型读取
MEM_KEY_WRITE = 12345  # (可选) 读取模型返回的动作

def main():
    try:
        # 1. 创建共享内存 (如果已存在则连接，否则创建)
        # 这里的 size 设大一点，防止溢出
        shm_state = sysv_ipc.SharedMemory(MEM_KEY_READ, flags=sysv_ipc.IPC_CREAT, mode=0o666, size=2048)
        print(f"环境模拟器启动成功! State Key: {MEM_KEY_READ}")
        
        # 尝试连接动作内存 (用于接收模型指令，看它有没有反应)
        shm_action = None
        print("等待动作内存创建...")

        counter = 0
        
        while True:
            counter += 1
            
            # --- 2. 构造 15 个模拟的 TCP 状态参数 ---
            # 顺序参考 envwrapper.py: 
            # [0]delay, [1]throughput, [2]samples, [3]delta_t, [4]target(0), [5]cwnd, 
            # [6]pacing, [7]loss, [8]srtt, [9]ssthresh, [10]pkts_out, [11]retrans, 
            # [12]max_pkts, [13]mss, [14]min_rtt
            
            # 模拟一些随机波动的数据
            d = random.uniform(0.01, 0.05)       # delay (s)
            thr = random.uniform(1000, 5000)     # throughput (bytes/s)
            samples = 10                         # samples
            delta_t = 0.02                       # time interval
            target = 0                           # unused
            cwnd = random.randint(10, 100)       # cwnd
            pacing = thr * 1.2                   # pacing rate
            loss = 0.0 if random.random() > 0.1 else 0.01 # 偶尔丢包
            srtt = d * 1000                      # srtt (ms)
            ssthresh = 1000                      # slow start threshold
            pkts_out = cwnd                      # packets out
            retrans = 0                          # retrans
            max_pkts = cwnd + 10                 # max packets
            mss = 1448                           # MSS
            min_rtt = srtt * 0.8                 # min rtt

            # 拼装成字符串列表
            params = [
                d, thr, samples, delta_t, target, cwnd, 
                pacing, loss, srtt, ssthresh, pkts_out, retrans, 
                max_pkts, mss, min_rtt
            ]
            
            # --- 3. 格式化协议 ---
            # 格式: "ID val1 val2 ... val15\0"
            # 加上 ID 刚好 16 个数
            msg_str = f"{counter}"
            for p in params:
                msg_str += f" {p:.4f}"
            msg_str += "\0"
            
            # --- 4. 写入共享内存 ---
            shm_state.write(msg_str.encode('utf-8'))
            
            # 打印发送日志 (每10次打印一次，防止刷屏)
            if counter % 10 == 0:
                # 如果之前没连接上，现在再试一次
                if shm_action is None:
                    try:
                        shm_action = sysv_ipc.SharedMemory(MEM_KEY_WRITE)
                        print(f"成功连接到动作内存 (Key: {MEM_KEY_WRITE})！")
                    except:
                        pass  # 还没创建，继续等待
                
                # 尝试看看模型有没有回传动作
                action_feedback = "等待模型..."
                if shm_action:
                    try:
                        raw = shm_action.read().decode('utf-8').split('\0')[0]
                        if raw:
                            # 格式 "ID Action"
                            parts = raw.split()
                            if len(parts) >= 2:
                                action_id = int(parts[0])
                                # 检查是否是最近几个ID的动作（因为可能有延迟）
                                if abs(action_id - counter) <= 5:
                                    action_feedback = f"收到动作 ID={action_id}: {parts[1]}"
                    except Exception as e:
                        pass  # 忽略读取错误
                
                print(f"[Env] 发送状态 ID={counter} | {action_feedback}")

            # 控制发送频率 (模拟 RTT，例如 50ms)
            time.sleep(0.05) 

    except KeyboardInterrupt:
        print("\n正在清理资源...")
        shm_state.remove()
        print("已删除共享内存。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()