/* 

Program to control green lights and trigger external (Photron) cameras
under control serial port (for use with flydra).

DDB0 - trigger source output to cameras

*/

#include <avr/io.h>
#include <avr/signal.h>
#include <avr/interrupt.h>
#include "ser169.h"

#define TEMPERATURE_SENSOR  0
#define FALSE 0
#define TRUE 1
void OSCCAL_calibration(void); /* forward decl. */
void Delay(unsigned int millisec); /* forward decl. */

// Macro definitions
//mtA - 
// sbi and cbi are not longer supported by the avr-libc
// to avoid version-conflicts the macro-names have been 
// changed to sbiBF/cbiBF "everywhere"
#define sbiBF(port,bit)  (port |= (1<<bit))   //set bit in port
#define cbiBF(port,bit)  (port &= ~(1<<bit))  //clear bit in port

#define ACTION_NONE 0
#define ACTION_GO   'g'
#define ACTION_STOP 's'
#define ACTION_GETTEMP 't'

// global vars
char ADC_temp_low =0;
char ADC_temp_high=0;

void ADC_read2(void)
{
    // To save power, the voltage over the LDR and the NTC is turned off when not used
    // This is done by controlling the voltage from a I/O-pin (PORTF3)
    sbiBF(PORTF, PF3); // mt sbi(PORTF, PORTF3);     // Enable the VCP (VC-peripheral)
    sbiBF(DDRF, DDF3); // sbi(DDRF, PORTF3);        

    sbiBF(ADCSRA, ADEN);     // Enable the ADC

    //do a dummy readout first
    ADCSRA |= (1<<ADSC);        // do single conversion
    while(!(ADCSRA & 0x10));    // wait for conversion done, ADIF flag active
        
    ADCSRA |= (1<<ADSC);        // do single conversion
    while(!(ADCSRA & 0x10));    // wait for conversion done, ADIF flag active
        
    ADC_temp_low = ADCL;            // read out ADCL register
    ADC_temp_high= ADCH;    // read out ADCH register        

    cbiBF(PORTF,PF3); // mt cbi(PORTF, PORTF3);     // disable the VCP
    cbiBF(DDRF,DDF3); // mt cbi(DDRF, PORTF3);  
    
    cbiBF(ADCSRA, ADEN);      // disable the ADC

}

void RTC_init(void)
{
  cli(); // mt __disable_interrupt();  // disabel global interrupt

  cbiBF(TIMSK2, TOIE2);             // disable OCIE2A and TOIE2
  
  ASSR = (1<<AS2);        // select asynchronous operation of Timer2
  
  TCNT2 = 0;              // clear TCNT2A
  TCCR2A |= (1<<CS22) | (1<<CS20);             // select precaler: 32.768 kHz / 128 = 1 sec between each overflow
  
  while((ASSR & 0x01) | (ASSR & 0x04));       // wait for TCN2UB and TCR2UB to be cleared
  
  TIFR2 = 0xFF;           // clear interrupt-flags
  sbiBF(TIMSK2, TOIE2);     // enable Timer2 overflow interrupt
  
  sei(); // mt __enable_interrupt();                 // enable global interrupt
}

SIGNAL(SIG_OVERFLOW2)
{
  // nothing
  //gSECONDS++; // increment second
}

void Initialization(void) {
  OSCCAL_calibration();

  ADMUX = TEMPERATURE_SENSOR;
  ADCSRA = (1<<ADEN) | (1<<ADPS1) | (1<<ADPS0);    // set ADC prescaler to , 1MHz / 8 = 125kHz    

  RTC_init();

  // USART
  UART_init();

  DDRB = ~(1<<DDB5); // all pins but 5 (piezo) are trigger output
}

int main(void) {
  unsigned char last_char_input;
  unsigned char do_action;
  int temperature;

  Initialization();

  PORTB |= ~(1<<DDB5); // set all trigger pins high

  while(1) {
    last_char_input = UART_Getchar();
    
    do_action = ACTION_NONE;
    
    switch (last_char_input) {
    case 'g':
      do_action = ACTION_GO;
      break;
    case 's':
      do_action = ACTION_STOP;
      break;
    case 't':
      do_action = ACTION_GETTEMP;
      break;
    default:
      break;
    }

    switch (do_action) {

    case ACTION_GO:
      UART_Putchar('g');
      break;

    case ACTION_STOP:
      UART_Putchar('s');
      break;

    case ACTION_GETTEMP:
      ADC_read2();
      UART_Putchar(ADC_temp_high);
      UART_Putchar(ADC_temp_low);
      break;      

    case ACTION_NONE:
    default:
      UART_Putchar('?');
      break;
      
    }
      
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

    // CLKPR = (1<<CLKPS1) | (1<<CLKPS0); // ADS - sets clock division factor to 8
    // ADS - thus, with calibrated internal crystal at 8 MHz, this gives us 1 MHz
    
    TIMSK2 = 0;             //disable OCIE2A and TOIE2

    ASSR = (1<<AS2);        //select asynchronous operation of timer2 (32.768kHz)
    /*
      Calculation of internal RC oscillator value - ADS

      f_xtal = external quartz crystal = 32768 Hz = 32.768 kHz
      ticks_xtal = compare value of timer2
      f_osc = internal RC oscillator frequency = 14745600 = 14.7456 MHz
      ticks_osc = f_osc/f_xtal*ticks_xtal
     */
#define ticks_xtal 60

    OCR2A = ticks_xtal;            // set timer2 compare value 

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

        if (temp > 27135)	  // ticks_osc + 0.5%
        {
            OSCCAL--;   // the internRC oscillator runs to fast, decrease the OSCCAL
        }
        else if (temp < 26865)	  // ticks_osc - 0.5%
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
