import cv2
from ultralytics import YOLO
import time

# ================= 报警参数配置 =================
# 根据 metadata.yaml 对齐的真实类别 ID
CLS_CLOSED_EYE = 0  # 闭眼
CLS_YAWN = 2        # 打哈欠

ALARM_THRES_EYE = 2.0   # 闭眼持续超过 2.0 秒触发报警
ALARM_THRES_YAWN = 3.0  # 打哈欠持续超过 3.0 秒触发报警

# 状态记录器（记录危险行为开始的时间点）
time_closed_eye_start = None
time_yawn_start = None
# ================================================

# 1. 加载 NCNN 模型
# 确保终端运行的当前目录下有 weights 文件夹
model_path = "./weights/drowsiness-best_ncnn_model"
model = YOLO(model_path, task="detect")

# 2. 初始化摄像头
cap = cv2.VideoCapture(0)
# 设置预览分辨率，保证树莓派 4 的流畅度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("--- 疲劳驾驶检测演示程序已启动（包含时序报警） ---")
print("提示：在 SSH 窗口按 Ctrl+C 停止，或在画面窗口按 'q' 退出")

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        print("无法读取摄像头画面，程序退出。")
        break

    # 3. NCNN 推理
    # imgsz=320 必须与导出时一致
    results = model.predict(frame, imgsz=320, conf=0.25, stream=True, verbose=False)

    for r in results:
        # 获取画好检测框的图像
        annotated_frame = r.plot()
        
        # 重置当前帧的状态标志
        is_eye_closed_now = False
        is_yawn_now = False

        # 遍历当前帧中检测到的所有目标
        for box in r.boxes:
            cls_id = int(box.cls[0].item()) # 获取类别的 ID
            
            if cls_id == CLS_CLOSED_EYE:
                is_eye_closed_now = True
            elif cls_id == CLS_YAWN:
                is_yawn_now = True

        # ================= 闭眼报警逻辑 =================
        if is_eye_closed_now:
            # 如果是刚刚闭眼，记录下当前时间
            if time_closed_eye_start is None:
                time_closed_eye_start = time.time()
            # 如果已经闭眼一段时间了，计算持续时间
            elif time.time() - time_closed_eye_start > ALARM_THRES_EYE:
                # 触发报警！在屏幕上打出显眼的红色字
                cv2.putText(annotated_frame, "ALARM: WAKE UP!!!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                print("\033[91m[危险] 检测到长时间闭眼！\033[0m")
        else:
            # 只要没检测到闭眼（比如检测到 Neutral 睁眼状态），计时器清零
            time_closed_eye_start = None

        # ================= 打哈欠报警逻辑 =================
        if is_yawn_now:
            if time_yawn_start is None:
                time_yawn_start = time.time()
            elif time.time() - time_yawn_start > ALARM_THRES_YAWN:
                cv2.putText(annotated_frame, "ALARM: YAWNING!!!", (50, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
                print("\033[93m[警告] 检测到长时间打哈欠！\033[0m")
        else:
            time_yawn_start = None

        # ================= 计算并显示 FPS =================
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 4. 显示结果
        cv2.imshow("Real-time Drowsiness Detection", annotated_frame)

    # 按下键盘上的 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()