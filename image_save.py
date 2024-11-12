import cv2
import os
import datetime

# 指定保存图像的文件夹路径
save_folder = 'captured_images'
os.makedirs(save_folder, exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按 's' 键保存当前画面，按 'q' 键退出程序")

while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    if not ret:
        print("无法接收画面，退出...")
        break

    # 显示画面
    cv2.imshow('Camera', frame)

    # 检测按键
    key = cv2.waitKey(1)
    if key == ord('s'):  # 按下 's' 键保存当前图像
        # 使用时间戳命名文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_folder, f"{timestamp}.png")

        # 保存图像
        cv2.imwrite(filename, frame)
        print(f"图像已保存: {filename}")
    elif key == ord('q'):  # 按下 'q' 键退出
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
