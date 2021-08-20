import socket
import re
import time

import gpio_control

HOST = '0.0.0.0'
PORT = 6668
BUF_SIZE = 1024

LED_SLEEP_SPAN = 0.1

emotions = []
emo_led_map = {0: 10, 1: 11, 2: 12}


def control_led_by_emotion(emo):
    gpio_control.set_all_leds('OFF')
    time.sleep(LED_SLEEP_SPAN)
    gpio_control.set_led(emo_led_map[emo_id], 'ON')


if __name__ == '__main__':
    gpio_control.reset_all_ports()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)  # 接收的连接数
    client, address = server.accept()  # 因为设置了接收连接数为1，所以不需要放在循环中接收
    while True:  # 循环收发数据包，长连接
        data = client.recv(BUF_SIZE)
        print(data.decode())  # python3 要使用decode
        # client.close() #连接不断开，长连接
        regex = re.compile(r"emo=(\d+)")
        emo_ids = re.findall(regex, data.decode())
        if len(emo_ids) == 0:
            continue
        elif len(emo_ids) == 1: # Normal Situation
            emo_id = emo_ids[0]
            control_led_by_emotion(emo_id)
        else: # Multiple Siuation
            pass