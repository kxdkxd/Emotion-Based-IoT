import socket
import re

PUBLIC_MAP_IP = "47.103.154.116"
PUBLIC_MAP_PORT = 6668

BUF_SIZE = 8

regex = re.compile(r"status=(\d+)")

global s


def setup_tcp_conn(ip=PUBLIC_MAP_IP, port=PUBLIC_MAP_PORT):
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP Connection
    s.connect((ip, port))


def send_emotion(emo_id):
    global s
    s.send(("emo=%s" % str(emo_id)).encode())


def recv_device_status():
    global s
    c = s.recv(BUF_SIZE)
    c = c.decode()
    li = re.findall(regex, c)
    if len(li) == 1:
        status = int(li[0])
        print("Recvd Status From RaspberryPi, status=%s" % status)
