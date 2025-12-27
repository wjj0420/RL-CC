import sysv_ipc
import time

# 必须与 run_inference.py 里的 MEM_KEY_WRITE 一致
MEM_KEY_WRITE = 12345 

def main():
    try:
        # 连接到模型写入动作的共享内存
        shm = sysv_ipc.SharedMemory(MEM_KEY_WRITE)
        print(f"成功连接到输出内存 (Key: {MEM_KEY_WRITE})")
        print("正在监听模型输出...")
        
        last_val = None
        
        while True:
            # 读取内存
            raw_bytes = shm.read()
            raw_str = raw_bytes.decode('utf-8').split('\0')[0]
            
            if not raw_str:
                continue
                
            # 只有当数据变化时才打印
            if raw_str != last_val:
                try:
                    step_id, action_val = raw_str.split()
                    print(f"Step: {step_id} -> Action: {action_val}")
                except:
                    pass # 忽略解析错误
                last_val = raw_str
            
            # 稍微休眠避免占用过高 CPU，监控不需要微秒级响应
            time.sleep(0.01)

    except sysv_ipc.ExistentialError:
        print("错误: 共享内存不存在。请确保 run_inference.py 正在运行！")
    except KeyboardInterrupt:
        print("\n停止监控。")

if __name__ == "__main__":
    main()



