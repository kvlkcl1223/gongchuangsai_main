import Jetson.GPIO as GPIO
import time


def is_gpio_low(pin):
    """
    检查指定 GPIO 引脚是否为低电平。

    参数:
    - pin: GPIO 引脚号 (根据 BCM 编号)

    返回:
    - True: 如果引脚是低电平
    - False: 如果引脚是高电平
    """
    GPIO.setmode(GPIO.BCM)  # 使用 BCM 引脚编号
    GPIO.setup(pin, GPIO.IN)  # 将引脚设置为输入模式

    state = GPIO.input(pin)  # 读取引脚状态
    GPIO.cleanup(pin)  # 清理引脚以释放资源

    return state == GPIO.LOW  # 返回是否为低电平


# 示例：检查 GPIO17 引脚是否为低电平
if __name__ == "__main__":
    gpio_pin = 7  # 修改为实际使用的 GPIO 编号
    while True:
        if is_gpio_low(gpio_pin):
            print(f"GPIO{gpio_pin} is LOW.")
        else:
            print(f"GPIO{gpio_pin} is HIGH.")
        time.sleep(1)  # 每隔 1 秒检查一次
