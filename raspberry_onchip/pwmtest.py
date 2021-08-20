import RPi.GPIO as gpio
import time


if __name__ == '__main__':
    GPIO_PORT_NUMBER = 2 # Connect the LED at GPIO 2
    PWM_FREQUENCY = 80
    gpio.setmode(gpio.BCM)
    gpio.setup(GPIO_PORT_NUMBER, gpio.OUT)
    pwm = gpio.PWM(GPIO_PORT_NUMBER, PWM_FREQUENCY)
    pwm.start(0)
    while True:
        pwm.ChangeDutyCycle(100)
        time.sleep(1)
    '''while True:
        print("Begin")
        for i in range(0, 100):
            pwm.ChangeDutyCycle(i)
            time.sleep(0.01)
        for i in reversed(range(0, 100+1)):
            pwm.ChangeDutyCycle(i)
            time.sleep(0.01)
        print("Done")'''
