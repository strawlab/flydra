#include "handler.h"

#include <avr/io.h>
#define F_CPU 16000000
#define HANDLER_FREQ(X) (F_CPU/256)

void isr(void);
void isr2(void);

int main(void)
{
  Handler_Init();

  DDRE=0xFF; // data direction register for port E = all output
  DDRF=0xFF; // data direction register for port F = all output

  PORTE=0x00;

  // n/(F_CPU/256) // period in seconds
  Reg_Handler(isr,313,0,1); // period = 5.xxx msec
  Reg_Handler(isr2,6250,1,1); // period = 100 msec

  while(1);

  return 0;
}

void isr(void) {
  PORTF=~PORTF;
}

void isr2(void) {
  PORTE=~PORTE;
}
