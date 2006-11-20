/* 

WARNING: OUTDATED! - ADS 10/14/06

Firmware operation:

Timer1 controls servo motors through PWM signals. These are output on
PORTB, pins PB5 and PB6. (PB7 should be reserved if another PWM signal
on the same carrier is needed.) The TOP value (with clock select)
controls the base frequency, while the output compare values set the
pulse width.

Timer3 generates external trigger pulses for the camera. The TOP value
(with clock select) set the interval, while output compare A sets the
pulse duration.

PB0 senses whether the camera is integrating or not. When integration
stops (PB0 transitions from high to low), PB1 is set high and timer0
is started. When timer0 reaches TOP, an overflow interrupt is
generated, and PB1 is set low. (This was not done using a PWM-like
mode because it will enable more flexible, software-based, use of
timers if further functions are required on the device.)

Summary:

| PB0 | in  | CAM INTEG (pin change interrupt 0) |
| PB1 | out | HEAT ON |
| PB5 | out | PWM A (timer1 output compare A) |
| PB6 | out | PWM B (timer1 output compare B) |

| PC6 | out | CAM TRIG (timer3 output compare A) |

USB control:

"ENDPOINT_BULK_OUT":
first 8 bytes (mostly timer1 control):
OCR1AH
OCR1AL
OCR1BH
OCR1BL

ICR1H
ICR1L
enter_DFU
clock_select_timer1

next 8 bytes:
new_timer3_data_and_clock_select_timer3
OCR3AH
OCR3AL
ICR3H

ICR3L
new_timer0_data_and_clock_select_timer0
timer0_start_val
reserved

*/