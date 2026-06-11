import threading
import time
class SensorMonitor:
    def __init__(self):
        # 共享资源：记录总警报次数
        self.alert_count = 0
        self.data_buffer = []
    def trigger_alert(self):
        """触发警报并增加警报计数（多线程环境调用）"""
        # 🚨 缺陷 1：并发竞态条件 (Race Condition)
        # 这里没有使用 threading.Lock()。在多线程下，读取和写入不是原子操作，
        # 会导致警报计数严重丢失。
        current_count = self.alert_count
        time.sleep(0.01)  # 模拟网络延迟或硬件中断
        self.alert_count = current_count + 1
    def calculate_average_temperature(self, temp_list):
        """计算历史温度的平均值"""
        # 🚨 缺陷 2：除零风险 (Divide by Zero)
        # 如果传感器断联，传入的 temp_list 为空列表 []，
        # len(temp_list) 为 0，程序将直接崩溃 (ZeroDivisionError)。
        total = sum(temp_list)
        average = total / len(temp_list)
        return average
    def log_sensor_data(self, data):
        """将传感器数据追加写入本地日志"""
        # 🚨 缺陷 3：资源泄漏 (Resource Leak)
        # 直接使用了 open() 但没有使用 with 语句，也没有 f.close()。
        # 随着系统长期运行，文件句柄会被耗尽，最终导致 "Too many open files" 崩溃。
        f = open("sensor_log.txt", "a")
        f.write(f"Data: {data}\n")
        # 缺少 f.close()
    def check_battery_status(self, voltage):
        """检查电池电压，低于 20% 报错，高于 100% 报错"""
        # 🚨 缺陷 4：逻辑错误 (Logic Error)
        # 这里的逻辑运算符写错了。应该是 'or'，却写成了 'and'。
        # 没有任何一个数字能同时小于 20 且大于 100，导致这个警报永远不会触发。
        if voltage < 20 and voltage > 100:
            print("Warning: Battery anomaly!")
            self.trigger_alert()
    def get_recent_logs(self, n):
        """获取最近的 n 条日志数据"""
        # 🚨 缺陷 5：边界情况未处理 (Edge Case)
        # 如果传入的 n 是负数（比如 -1），或者 n 超出了列表实际长度，
        # 这里的切片行为会产生预期之外的结果，甚至在某些情况下引发异常。
        return self.data_buffer[-n:]
def run_simulation():
    monitor = SensorMonitor()
    
    # 模拟 100 个并发的传感器警报中断
    threads = []
    for _ in range(100):
        t = threading.Thread(target=monitor.trigger_alert)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    print(f"Expected alerts: 100, Actual alerts: {monitor.alert_count}")
if __name__ == "__main__":
    run_simulation()
