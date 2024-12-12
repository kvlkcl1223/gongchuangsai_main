# -*- coding: utf-8 -*-
import serial
import time
def open_serial(port, baudrate, timeout=None, retry_interval=1):
    """
    尝试打开串口，直到成功为止。

    :param port: 串口端口号，例如 'COM3' 或 '/dev/ttyUSB0'
    :param baudrate: 波特率，例如 9600
    :param timeout: 超时时间，默认 None
    :param retry_interval: 重试间隔时间（秒），默认 1 秒
    :return: 打开的 serial.Serial 对象
    """
    while True:
        try:
            ser = serial.Serial(port, baudrate, timeout=timeout)
            print(r"Successfully opened serial port:")
            return ser  # 返回成功打开的串口对象
        except serial.SerialException as e:
            print(r"Failed to open . Retrying in second(s)...")
            time.sleep(retry_interval)


port = '/dev/ttyUSB0'  # 替换为你的串口号
baudrate = 115200
timeout = 1
ser = open_serial(port=port, baudrate=baudrate, timeout=timeout, retry_interval=1)
ser.reset_input_buffer()
buffer = ""

while True:
    while True:
        try:
            if ser.in_waiting > 0:  # 检查是否有数据等待读取
                # 读取一行数据并解码
                received_data = ser.readline().decode('ascii').strip()
                print('received_data', received_data)
        except Exception as e:
            print(f'Unexpected error:')
            ser.close()
            time.sleep(0.2)  # 程序暂停一秒后重试
            ser = open_serial(port=port, baudrate=baudrate, timeout=timeout, retry_interval=1)
            print('已重新打开')