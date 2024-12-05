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
            print(f"Successfully opened serial port: {port}")
            return ser  # 返回成功打开的串口对象
        except serial.SerialException as e:
            print(f"Failed to open {port}: {e}. Retrying in {retry_interval} second(s)...")
            time.sleep(retry_interval)


port = '/dev/ttyTHS1'  # 替换为你的串口号
baudrate = 115200
timeout = 1
ser = open_serial(port=port, baudrate=baudrate, timeout=timeout, retry_interval=1)
ser.reset_input_buffer()


while True:
    ser.write("Tar=q2!".encode('ascii'))
    time.sleep(9)