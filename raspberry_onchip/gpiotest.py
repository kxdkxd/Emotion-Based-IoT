import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
for i in range(0,7+1):
    pin = i
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(0.1)

#GPIO.cleanup()
