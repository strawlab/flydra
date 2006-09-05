#include <avr/io.h>
#include <avr/signal.h>
#include <avr/interrupt.h>


int main(void) {
  ASSR = (1<<AS2);        //select asynchronous operation of timer2 (32.768kHz)
  OCR2A = 0x7F; // output compare value, halfway between top and bottom (arbitrary)
  TCCR2A = (0<<COM2A1)|(1<<COM2A0)|(0<<CS22)|(0<<CS21)|(1<<CS20); // toggle pin on OCR match, clock prescalar=8
  DDRB = ~(1<<DDB5); // all pins but 5 (piezo) are trigger output
  while(1) {
  }
}  
