import socket
import re
import time

import gpio_control

HOST = '0.0.0.0'
PORT = 6668
BUF_SIZE = 1024

LED_SLEEP_SPAN = 0.01

emotions = []
emo_led_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}


def control_led_by_emotion(emo):
    gpio_control.set_all_leds('OFF')
    time.sleep(LED_SLEEP_SPAN)
    gpio_control.set_led(emo_led_map[emo], 'ON')


if __name__ == '__main__':
    gpio_control.reset_all_ports(*list(emo_led_map.values()))
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    client, address = server.accept()
    while True:
        data = client.recv(BUF_SIZE)
        print(data.decode())
        # client.close() 
        regex = re.compile(r"emo=(\d+)")
        emo_ids = re.findall(regex, data.decode())
        if len(emo_ids) == 0:
            continue
        elif len(emo_ids) == 1: # Normal Situation
            emo_id = int(emo_ids[0])
            control_led_by_emotion(emo_id)
        else: # Multiple Siuation
            pass