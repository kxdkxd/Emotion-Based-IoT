import RPi.GPIO as gpio


PWM_FREQUENCY = 80  # Larger value can reduce the blinking of LED, however more CPU resources will cost.
DO_FLASH_LED = False  # Only falsh the LED once, if level reaches the threshold.
DO_RESPIRATION = False  # Make a gradual change in lighting depends on attention level.

def reset_all_ports(*ports):
    for port in ports:
        init_gpio_out(port)
    set_all_leds('OFF', *ports)


def set_all_leds(state, *ports):
    state = state.upper()
    for port in ports:
        _set_led(port, state)


def set_led(gpio_port_number, state, pwm_duty_cycle=80):
    set_all_leds('OFF', 0,1,2,3,4,5,6,7)
    _set_led(gpio_port_number, state, pwm_duty_cycle)


def _set_led(gpio_port_number, state, pwm_duty_cycle=80):
    if state.upper() == 'ON':
        if DO_RESPIRATION:
            pwm.ChangeDutyCycle(pwm_duty_cycle)  # Write PWM duty.
        else:
            gpio.output(gpio_port_number, gpio.HIGH)
    else:
        gpio.output(gpio_port_number, gpio.LOW)


def init_gpio_out(gpio_port_number):  # Setup GPIO to control LED. The positive part locates at given channel, negative part locates at GND.
    gpio.setmode(gpio.BCM)
    gpio.setup(gpio_port_number, gpio.OUT)
    if DO_FLASH_LED or DO_RESPIRATION:  # Setup PWM module if needed.
        global pwm
        pwm = gpio.PWM(gpio_port_number, PWM_FREQUENCY)
        pwm.start(0)
