#include <avr/io.h>
#include <avr/signal.h>
#include <avr/interrupt.h>
#ifdef USELCD
#include "LCD_driver.h"
#include "LCD_functions.h"
#endif
#include "blink.h"

//#define LIGHTS_ON '1'
//#define LIGHTS_OFF_LONG '0'
//#define LIGHTS_OFF_050MSEC 'R'
//#define LIGHTS_OFF_100MSEC 'S'
//#define LIGHTS_OFF_250MSEC 'T'
//#define LIGHTS_FOCAL_OFF_050MSEC 'A'
//#define LIGHTS_FOCAL_OFF_100MSEC 'B'

#define MSG_STATE_READY 0
#define MSG_STATE_READ_DUR 1
#define ACTION_NONE 'a'
#define ACTION_TRIGGER 'b'
#define ACTION_DARK 'c'
#define ACTION_LIGHT 'd'
#define ACTION_TIMED_DARK 'e'

void delayms(unsigned int millisec)
{
	uint8_t i;

	while (millisec--) {
	  for (i=0; i<199; i++) { // ads calibrated 2006 03 05
			asm volatile ("nop"::);
		}
	}
}

void USART_init(void) {
  unsigned int ubrr;

  // Set baud rate
  ubrr = 12; // 4800 bps, see atmega169 manual pg. 174
  UBRRH = (unsigned char)(ubrr>>8);
  UBRRL = (unsigned char)ubrr;

  // Enable receiver and transmitter
  UCSRB = (1<<RXEN) | (1<<TXEN);
  
  // Set frame format 8N1
  // UCSRC defaults to 8N1 = (3<<UCSZ0)

  // Set frame format 7E1
  //UCSRC = (1<<UPM1)|(1<<UCSZ1);
}


void Initialization(void) {
  OSCCAL_calibration();

#ifdef USELCD
  LCD_Init();
#endif

  // USART
  USART_init();

  DDRB = (1<<DDB0) | (1<<DDB2) | (1<<DDB4); // output pins enabled
  PORTB |= ((1<<DDB0) | (1<<DDB2)); // turn lights on

  // timer2
  ASSR = (1<<AS2);        //select asynchronous operation of timer2 (32,768kHz)
  TIMSK2=(1<<OCIE2A); // enable interrupt
  TCCR2A = 0; //stop timer2

  // timer0
  TIMSK0=(1<<TOIE0); // enable interrupt
  TCCR0A = 0; //stop timer0
}

SIGNAL(SIG_OUTPUT_COMPARE2)
{

  PORTB |= ((1<<DDB0) | (1<<DDB2)); // turn lights on
  TCCR2A = 0; //stop timer2
#ifdef USELCD
  LCD_puts("t over", SCROLLMODE_LOOP);
#endif

}

SIGNAL(SIG_OVERFLOW0)
{
  PORTB &= ~(1<<DDB4); // turn off trigger pin
  TCCR0A = 0; //stop timer0
}

int main(void) {
  unsigned char msg_state;

  unsigned char last_char_input;
  unsigned char do_action;

  unsigned char CH,CL;
    
  msg_state=MSG_STATE_READY;
  Initialization();
#ifdef USELCD
  LCD_puts("READY", SCROLLMODE_LOOP);
#endif
  while(1) {
    
    while (!(UCSRA & (1<<RXC))) {} // wait until USART received byte
    last_char_input = UDR; // get byte
    do_action = ACTION_NONE;
    
    switch (msg_state) {
    case MSG_STATE_READY:
      switch (last_char_input) {
      case 'X':
	do_action = ACTION_TRIGGER;
	break;
      case '0':
	do_action = ACTION_DARK;
	break;
      case '1':
	do_action = ACTION_LIGHT;
	break;
      case 't':
	msg_state = MSG_STATE_READ_DUR;
	break;
      default:
	break;
      }
      break;
    case MSG_STATE_READ_DUR:
      do_action = ACTION_TIMED_DARK;
      break;
    default:
      break;
    }

    switch (do_action) {

    case ACTION_TRIGGER:
#ifdef USELCD
      LCD_puts("trig", SCROLLMODE_LOOP);
#endif
      PORTB |= (1<<DDB4); // turn on trigger pin

      TCNT0 = 0;              // clear timer0 counter
      TCCR0A = (1<<CS00);     // start timer0 (no prescaling)
      break;


    case ACTION_DARK:
      TCCR2A = 0; //stop timer2
      PORTB &= ~((1<<DDB0) | (1<<DDB2)); // lights out
#ifdef USELCD
      LCD_puts("dark", SCROLLMODE_LOOP);
#endif
      break;


    case ACTION_LIGHT:
      TCCR2A = 0; //stop timer2
      PORTB |= ((1<<DDB0) | (1<<DDB2)); // turn lights on
#ifdef USELCD
      LCD_puts("light", SCROLLMODE_LOOP);
#endif
      break;


    case ACTION_TIMED_DARK:
#ifdef USELCD
      LCD_puts("ATD", SCROLLMODE_LOOP);
#endif

      msg_state = MSG_STATE_READY;


      OCR2A = (last_char_input-1);    // set timer2 compare value (for
				      // some reason, it takes 1 tick
				      // extra, so subtract that)

      TCNT2 = 0;                      // clear timer2 counter, (do
				      // after setting ocr2a because
				      // compare is blocked for one
				      // clock after setting)

      TCCR2A = (1<<CS22) | (1<<CS21) | (1<<CS20);     // start timer2
						      // with
						      // prescaling by
						      // 1024 (WGM is
						      // 0 = normal)
      
      PORTB &= ~((1<<DDB0) | (1<<DDB2)); // lights out

#ifdef USELCD
      LCD_puts("T ?", SCROLLMODE_LOOP);
#endif
      break;


    case ACTION_NONE:
    default:
#ifdef USELCD
      LCD_puts("????", SCROLLMODE_LOOP);
#endif
      break;

    }
      
    UDR=last_char_input; // echo (transmit) back on USART
  }
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

    CLKPR = (1<<CLKPS1) | (1<<CLKPS0); // ADS - sets clock division factor to 8
    // ADS - thus, with calibrated internal crystal at * MHz, this gives us 1 MHz
    
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
