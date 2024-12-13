# -*- coding: utf-8 -*-



import serial
import time
def open_serial_command(port):
    start_time = time.time()
    while time.time()-start_time <10:
        try:
            ser = serial.Serial(
                port="/dev/ttyUSB0",
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            print(f"Successfully opened serial port: {port}")
            return ser  # 返回成功打开的串口对象
        except serial.SerialException as e:
            print(f"Failed to open {port}: {e}. Retrying in {0.1} second(s)...")
            time.sleep(0.1)


ser_command = open_serial_command("/dev/ttyUSB0")
ser_command.reset_input_buffer()
buffer = ""

try:
    if ser_command.in_waiting > 0:  # 检查是否有数据等待读取
        # 读取一行数据并解码
        try:
            received_data = ser_command.readline().decode('ascii').strip()
            print("received_data", received_data)
            buffer += received_data  # 将接收到的数据添加到缓冲区

            # 假设数据以特定标识符结束（例如"!"）
            if '!' in buffer:
                messages = buffer.split('!')  # 根据标识符分割消息
                for message in messages:
                    if message:  # 确保消息不为空
                        print(f"command接收到的数据: {message}")
                        if message == "com=q1":  # 替换为实际的条件
                            print("Tar=q1!")
                            print('可回收垃圾=!')
                        elif message == "com=q2":  # 替换为实际的条件
                            print("Tar=q2!")
                            print('有害垃圾=!')
                        elif message == "com=q3":  # 替换为实际的条件
                            print("Tar=q3!")
                            print('厨余垃圾=!')
                        elif message == "com=q4":  # 替换为实际的条件
                            print("Tar=q4!")
                            print('其他垃圾=!')
                buffer = ""  # 清空缓冲区
        except UnicodeDecodeError:
            # 如果解码失败，处理异常
            # queue_transmit.put("Tar=repeat!")
            print("Decoding error: received data contains invalid ASCII characters.")

except Exception as e:
    print(f"Unexpected error: {e}")
    ser_command.close()
    time.sleep(0.2)  # 程序暂停一秒后重试
    ser_command = open_serial_command(port="/dev/ttyUSB0")
    print("已重新打开")

# while True:
#     while True:
#         try:
#             if ser.in_waiting > 0:  # 检查是否有数据等待读取
#                 print("ok")
#                 # 读取一行数据并解码
#                 received_data = ser.readline().decode('ascii').strip()
#                 print('received_data', received_data)
#         except Exception as e:
#             print('Unexpected error:')
#             ser.close()
#             time.sleep(0.2)  # 程序暂停一秒后重试
#             ser = serial.Serial(
#                 port="/dev/ttyUSB0",
#                 baudrate=115200,
#                 bytesize=serial.EIGHTBITS,
#                 parity=serial.PARITY_NONE,
#                 stopbits=serial.STOPBITS_ONE,
#             )
#             print('已重新打开')
#
#         time.sleep(0.1)