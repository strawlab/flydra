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
void Delay(uint32_t millisec); /* forward decl. */

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
#define ACTION_GETTIME 'T'
#define ACTION_GETFREQ 'f'

// global vars
volatile uint8_t gSUBSECTICKS=0;

volatile uint32_t gTIME_HIGH=0;
volatile uint8_t gTIME_LOW=0;

typedef struct timespec timespec;
struct timespec {
  uint32_t sec_high;
  uint8_t sec_low;
  uint8_t ticks_high;
  uint8_t ticks_med;
  uint8_t ticks_low;
};

/* 

Design:

System clock
============

timer2 increments with external crystal with a prescalar dividing it
by 128, and it 8 bits wide. Thus, it rolls over (and generates an
interrupt) once a second. (gTIME_HIGH*0x10+gTIME_LOW) counts how
many seconds have elapsed since the power was cycled.

timer1 increments at the system clock frequency (f_osc), which is
14.7456MHz. It is 16 bits wide and thus rolls over about 225 times per
second. (14745600./0xFFFF=225.003) gSUBSECTICKS keeps track of how
often this has occurred so far during the current second. Timer1 can
provide the finer detail of the current time.

*/

void send_freq_packet( ) {
  //14.7456 MHz
  UART_Putchar('f');
  UART_Putchar(4);
  UART_Putchar(0x00);
  UART_Putchar(0xe1);
  UART_Putchar(0x00);
  UART_Putchar(0x00);
  UART_Putchar('X');
}

void send_temperature_packet( uint16_t temperature ) {
  uint8_t tmp;

  UART_Putchar('t');
  UART_Putchar(2);
  tmp = (temperature>>8);
  UART_Putchar(tmp);
  tmp = (temperature&0xFF);
  UART_Putchar(tmp);
  UART_Putchar('X');
}

void send_timestamp_packet( timespec t ) {
  uint8_t tmp;

  UART_Putchar('T');
  UART_Putchar(8);

  tmp = (t.sec_high>>24) & 0xFF;
  UART_Putchar(tmp);
  tmp = (t.sec_high>>16) & 0xFF;
  UART_Putchar(tmp);
  tmp = (t.sec_high>>8) & 0xFF;
  UART_Putchar(tmp);
  tmp = t.sec_high & 0xFF;
  UART_Putchar(tmp);

  UART_Putchar(t.sec_low);
  UART_Putchar(t.ticks_high);
  UART_Putchar(t.ticks_med);
  UART_Putchar(t.ticks_low);
  UART_Putchar('X');
}

timespec get_time(void) {
  timespec result;
  uint8_t b2,b1,b0; // tmp copies are just registers

  cli();
  // first grab timer1 low then high (temp variables speed assembly)
  b0 = TCNT1L;
  b1 = TCNT1H;
  b2 = gSUBSECTICKS;

  result.sec_high=gTIME_HIGH;
  result.sec_low=gTIME_LOW;
  sei();

  result.ticks_low = b0; // copy from register (to stack?)
  result.ticks_med = b1;
  result.ticks_high = b2;

  return result;
}

uint16_t ADC_read(void)
{
  uint16_t result;
  uint8_t tmp0, tmp1;

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
        
    tmp0 = ADCL;            // read out ADCL register
    tmp1 = ADCH;    // read out ADCH register        

    result = tmp1*0x10 + tmp0;

    cbiBF(PORTF,PF3); // mt cbi(PORTF, PORTF3);     // disable the VCP
    cbiBF(DDRF,DDF3); // mt cbi(DDRF, PORTF3);  
    
    cbiBF(ADCSRA, ADEN);      // disable the ADC
    return result;
}

void RTC_init(void)
{
  cli(); // mt __disable_interrupt();  // disable global interrupt

  // START TIMER2

  cbiBF(TIMSK2, TOIE2);             // disable OCIE2A and TOIE2
  
  ASSR = (1<<AS2);        // select asynchronous operation of Timer2
  
  TCNT2 = 0;              // clear TCNT2A
  TCCR2A |= (1<<CS22) | (1<<CS20);             // select precaler: 32.768 kHz / 128 = 1 sec between each overflow
  
  while((ASSR & 0x01) | (ASSR & 0x04));       // wait for TCN2UB and TCR2UB to be cleared
  
  TIFR2 = 0xFF;           // clear interrupt-flags
  sbiBF(TIMSK2, TOIE2);     // enable Timer2 overflow interrupt
  
  // START TIMER1
  TCNT1H = 0;     // clear timer1 counter
  TCNT1L = 0;
  TCCR1B = (1<<CS10);     // start timer1 with no prescaling

  sei(); // mt __enable_interrupt();                 // enable global interrupt
}

SIGNAL(SIG_OVERFLOW1)
{
  gSUBSECTICKS++;
}

SIGNAL(SIG_OVERFLOW2)
{
  // reset timer1 so it was 0 when timer2 overflowed
  TCNT1H = 0;     // clear timer1 counter, write high byte first
  TCNT1L = 21;    // set time to number of clock cycles (as computed from looking at assembly) since timer2 overflow
  sei();  // allow the rest to be interruptable
  gSUBSECTICKS=0; // do this after enabling interrupts to clear any results of timer1 overflow during this interrupt

  if (gTIME_LOW==0xFF) {
    gTIME_LOW=0;
    gTIME_HIGH++;
  } else {
    gTIME_LOW++;
  }
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

  Initialization();

  PORTB |= ~(1<<DDB5); // set all trigger pins high

  while(1) {
    last_char_input = UART_Getchar();
    
    do_action = ACTION_NONE;
    
    switch (last_char_input) {
    case 'g': do_action = ACTION_GO;      break;
    case 's': do_action = ACTION_STOP;    break;
    case 'f': do_action = ACTION_GETFREQ; break;
    case 't': do_action = ACTION_GETTEMP; break;
    case 'T': do_action = ACTION_GETTIME; break;
    default:                              break;
    }

    switch (do_action) {

    case ACTION_GO:
      UART_Putchar('g');
      break;

    case ACTION_STOP:
      UART_Putchar('s');
      break;

    case ACTION_GETFREQ:
      send_freq_packet();
      break;      

    case ACTION_GETTEMP:
      send_temperature_packet(ADC_read());
      break;      

    case ACTION_GETTIME:
      send_timestamp_packet(get_time());
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
    uint8_t calibrate = FALSE;
    uint16_t temp;
    uint8_t tempL;

    // CLKPR = (1<<CLKPCE);        // set Clock Prescaler Change Enable

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

    Delay(16000);    // wait for external crystal to stabilise
    
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

        if (temp > 27135)	  // ticks_osc + slop
        {
            OSCCAL--;   // the internRC oscillator runs to fast, decrease the OSCCAL
        }
        else if (temp < 26865)	  // ticks_osc - slop
        {
            OSCCAL++;   // the internRC oscillator runs to slow, increase the OSCCAL
        }
        else
            calibrate = TRUE;   // the interRC is correct

        TCCR1B = (1<<CS10); // start timer1
    }
}

void Delay(uint32_t duration)
     // duration depends on clock speed
{
  uint8_t i;

  while (duration--) {
    for (i=0; i<125; i++) {
      asm volatile ("nop"::);
    }
  }
}
