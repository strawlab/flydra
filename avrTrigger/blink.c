#include <avr/io.h>
#include <avr/interrupt.h>
#include "blink.h"

#define LIGHTS_ON 0
#define LIGHTS_OFF 1
#define LIGHTS_BLINK 2

volatile char gState  = LIGHTS_ON;

void delayms(unsigned int millisec)
{
	// mt, int i did not work in the simulator:  int i; 
	uint8_t i;

	while (millisec--) {
	  for (i=0; i<169; i++) { // ads calibrated 2006 02 24
			asm volatile ("nop"::);
		}
	}
}

void Initialization(void) {
  OSCCAL_calibration();
}


int main(void) {
  Initialization();
  //  DDRB = 0xFF; // port B is all output
  sbiBF(DDRB,0); // B0 is output
  cbiBF(DDRB,2); // B2 is input
  cbiBF(DDRB,4); // B2 is input
  while(1) {
    if (gState==LIGHTS_ON) {
      sbiBF(PORTB, 0);
    } else {
      if (gState==LIGHTS_OFF) {
	cbiBF(PORTB, 0);
      } else {
	if (gState==LIGHTS_BLINK) {
	  // toggle port B
	  cbiBF(PORTB, 0);
	  delayms(50); // off 50 msec
	  sbiBF(PORTB, 0);
	  delayms(5); // on 5 msec
	}
      }

    }

    if((PINB & (1<<PINB2))) {
      // pins float high

      // b2 high
      if ((PINB & (1<<PINB4))) {
	// b4 high
	gState=LIGHTS_ON;
      }
      else {
	// b4 low
	gState=LIGHTS_OFF;
      }
    } else {
      // b2 low
      gState=LIGHTS_BLINK;
    }

  }
  return 0;
}




/*****************************************************************************
*
*   Function name : OSCCAL_calibration
*
*   Returns :       None
*
*   Parameters :    None
*
*   Purpose :       Calibrate the internal OSCCAL byte, using the external 
*                   32,768 kHz crystal as reference
*
*****************************************************************************/
void OSCCAL_calibration(void)
{
    unsigned char calibrate = FALSE;
    int temp;
    unsigned char tempL;

    CLKPR = (1<<CLKPCE);        // set Clock Prescaler Change Enable
    // set prescaler = 8, Inter RC 8Mhz / 8 = 1Mhz
    CLKPR = (1<<CLKPS1) | (1<<CLKPS0);
    
    TIMSK2 = 0;             //disable OCIE2A and TOIE2

    ASSR = (1<<AS2);        //select asynchronous operation of timer2 (32,768kHz)
    
    OCR2A = 200;            // set timer2 compare value 

    TIMSK0 = 0;             // delete any interrupt sources
        
    TCCR1B = (1<<CS10);     // start timer1 with no prescaling
    TCCR2A = (1<<CS20);     // start timer2 with no prescaling

    while((ASSR & 0x01) | (ASSR & 0x04));       //wait for TCN2UB and TCR2UB to be cleared

    Delay(1000);    // wait for external crystal to stabilise
    
    while(!calibrate)
    {
        cli(); // mt __disable_interrupt();  // disable global interrupt
        
        TIFR1 = 0xFF;   // delete TIFR1 flags
        TIFR2 = 0xFF;   // delete TIFR2 flags
        
        TCNT1H = 0;     // clear timer1 counter
        TCNT1L = 0;
        TCNT2 = 0;      // clear timer2 counter
           
        // shc/mt while ( !(TIFR2 && (1<<OCF2A)) );   // wait for timer2 compareflag    
        while ( !(TIFR2 & (1<<OCF2A)) );   // wait for timer2 compareflag

        TCCR1B = 0; // stop timer1

        sei(); // __enable_interrupt();  // enable global interrupt
    
        // shc/mt if ( (TIFR1 && (1<<TOV1)) )
        if ( (TIFR1 & (1<<TOV1)) )
        {
            temp = 0xFFFF;      // if timer1 overflows, set the temp to 0xFFFF
        }
        else
        {   // read out the timer1 counter value
            tempL = TCNT1L;
            temp = TCNT1H;
            temp = (temp << 8);
            temp += tempL;
        }
    
        if (temp > 6250)
        {
            OSCCAL--;   // the internRC oscillator runs to fast, decrease the OSCCAL
        }
        else if (temp < 6120)
        {
            OSCCAL++;   // the internRC oscillator runs to slow, increase the OSCCAL
        }
        else
            calibrate = TRUE;   // the interRC is correct

        TCCR1B = (1<<CS10); // start timer1
    }
}

/*****************************************************************************
*
*   Function name : Delay
*
*   Returns :       None
*
*   Parameters :    unsigned int millisec
*
*   Purpose :       Delay-loop
*
*****************************************************************************/
void Delay(unsigned int millisec)
{
	// mt, int i did not work in the simulator:  int i; 
	uint8_t i;

	while (millisec--) {
		for (i=0; i<125; i++) {
			asm volatile ("nop"::);
		}
	}
}
