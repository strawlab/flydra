#include "handler.h"

#include <avr/io.h>

void isr(void); // forward definition

int main(void)
{
  Handler_Init();

  DDRB=0xFF; // data direction register for port B = all output

  Reg_Handler(isr,31300,0,1); // period = 5.xxx msec

  while(1);

  return 0;
}

void isr(void) {
  PORTB=~PORTB;
}
