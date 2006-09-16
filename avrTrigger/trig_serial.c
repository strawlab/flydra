/* 

Theory of operation:

timer1 is the lowest 16 bits of the "master clock" and counts time
accordingly.

timer2 is used for the trigger output (because its interrupts are
processed first).

*/

#include <avr/io.h>
#include <avr/signal.h>
#include <avr/interrupt.h>
#include "ser169.h"

#define TEMPERATURE_SENSOR  0
#define FALSE 0
#define TRUE 1

// Macro definitions
// sbi and cbi are not longer supported by the avr-libc
// to avoid version-conflicts the macro-names have been 
// changed to sbiBF/cbiBF "everywhere"
#define sbiBF(port,bit)  (port |= (1<<bit))   //set bit in port
#define cbiBF(port,bit)  (port &= ~(1<<bit))  //clear bit in port

#define PT_STATE 'r'
#define PT_FREQ 'f'
#define PT_TEMP 't'
#define PT_TRIG_TS 'T'
#define PT_QUERY_TS 'Q'

#define ACTION_NONE 0
#define ACTION_GO   'g'
#define ACTION_STOP 's'
#define ACTION_GETTEMP 't'
#define ACTION_GETTIME 'T'
#define ACTION_GETFREQ 'f'

// output compare value, halfway between top and bottom (arbitrary)
#define TIMER2_COMPARE 0x7F

// calibrated on oscilloscope to give 10.0 msec interval
#define TRIG_COUNT_PERIOD 625

// make rate come to exactly (32 = 100.0 Hz, 16 = 200 Hz)
// IMPORTANT: make sure TRIG_COUNT_REMAINDER is less than TIMER2_COMPARE

// Must be uint8
#define TRIG_COUNT_REMAINDER 22

//#define DEBUGTRIG

// set to half of TRIG_COUNT_PERIOD for 50/50 duty cycle, set to 1 for minimal duration
#define TRIG_COUNT_CMP 10

// global vars
volatile uint8_t gSUBSECTICKS=0;

volatile uint32_t gTIME_HIGH=0;
volatile uint8_t gTIME_LOW=0;

uint16_t trig_count=0;
uint8_t doing_remainder=0;

typedef struct timespec timespec;
struct timespec {
  uint32_t sec_high;
  uint8_t sec_low;
  uint8_t ticks_high;
  uint8_t ticks_med;
  uint8_t ticks_low;
};

volatile uint8_t new_trigger_timestamp=0;
volatile timespec trigger_timestamp;

void send_state_packet(uint8_t val) {
  UART_Putchar(PT_STATE);
  UART_Putchar(0x01);
  UART_Putchar(val);
  UART_Putchar('X');
}

void send_freq_packet( void ) {
  //16.0 MHz
  UART_Putchar(PT_FREQ);
  UART_Putchar(4);
  UART_Putchar(0x00);
  UART_Putchar(0xf4);
  UART_Putchar(0x24);
  UART_Putchar(0x00);
  UART_Putchar('X');
}

void send_temperature_packet( uint16_t temperature ) {
  uint8_t tmp;

  UART_Putchar(PT_TEMP);
  UART_Putchar(2);
  tmp = (temperature>>8);
  UART_Putchar(tmp);
  tmp = (temperature&0xFF);
  UART_Putchar(tmp);
  UART_Putchar('X');
}

void send_timestamp_packet( unsigned char packet_type, timespec t ) {
  uint8_t tmp;

  UART_Putchar(packet_type);
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

void timer1_init(void)
{
  // START TIMER1
  TCNT1H = 0;     // clear timer1 counter
  TCNT1L = 0;
  TCCR1B = (1<<CS10);     // start timer1 with no prescaling

  TIMSK1 = (1<<TOIE1); // timer overflow interrupt enable

}

SIGNAL(SIG_OVERFLOW2) {
  // timer2
  if (trig_count == TRIG_COUNT_CMP) {
    //sbiBF(TCCR2A,COM2A0); // next output compare sets trigger
#ifdef DEBUGTRIG
    UART_Putchar('!');
#endif
  } else {
    if (trig_count == 0) {
      //cbiBF(TCCR2A,COM2A0); // next output compare clears trigger
      PORTB = 0x00;
      trig_count = TRIG_COUNT_PERIOD; // reset counter
    }
    // do nothing
  }
  trig_count--;
}

SIGNAL(SIG_OUTPUT_COMPARE2) {
  // timer2
  if (trig_count==0) {
    if (doing_remainder==0) {
      // first pass through
      TCNT2 -= TRIG_COUNT_REMAINDER;
      doing_remainder=1;
      return;
    } else {
      // second pass through
      doing_remainder=0;
    }
  } 

  if (trig_count==(TRIG_COUNT_CMP-1)) {
    PORTB = 0xFF;
    trigger_timestamp = get_time();
    new_trigger_timestamp = 1;
  }
}

SIGNAL(SIG_OVERFLOW1)
{
  //PORTB = 0xFF;
  if (gSUBSECTICKS==0xFF) {
    gSUBSECTICKS=0;
    if (gTIME_LOW==0xFF) {
      gTIME_LOW=0;
      gTIME_HIGH++;
    } else {
      gTIME_LOW++;
    }
  }
  else {
    gSUBSECTICKS++;
  }
}

void timer2_init(void) {
  // user's guide says to make sure to perform before setting DDRB
  OCR2A = TIMER2_COMPARE;
  TIMSK2 = (1<<OCIE2A)|(1<<TOIE2); // ouput compare interrupt enable, overflow interrupt enable
  //TIMSK2 = (1<<TOIE2); // timer overflow interrupt enable
}

void Initialization(void) {
  ADMUX = TEMPERATURE_SENSOR;
  ADCSRA = (1<<ADEN) | (1<<ADPS2) | (1<<ADPS1) | (1<<ADPS0);    // set ADC prescaler to 128, 16MHz / 128 = 125kHz    

  timer1_init();

  // USART
  UART_init();

  timer2_init();

  DDRD = 0x05; // for LEDs
  //DDRD = 0xFF; // for LEDs
  PORTD = 0x01; // red

  DDRB = ~(1<<DDB5); // all pins but 5 (piezo) are trigger output
}

int main(void) {
  unsigned char last_char_input;
  unsigned char do_action;

  Initialization();

  //PORTB |= ~(1<<DDB5); // set all trigger pins high
  PORTB = 0;

  while(1) {

    if (new_trigger_timestamp) {
      send_timestamp_packet(PT_TRIG_TS,trigger_timestamp);
      new_trigger_timestamp=0;
    }

    if (UART_CharReady()) {
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
	// start timer2
	//TCCR2A = (1<<COM2A1)|(0<<COM2A0)|(0<<CS22)|(0<<CS21)|(1<<CS20); // clear pin on OCR match, clock prescalar=8
	TCCR2A = (0<<COM2A1)|(0<<COM2A0)|(0<<CS22)|(0<<CS21)|(1<<CS20); // clear pin on OCR match, clock prescalar=8
#ifdef DEBUGTRIG
	UART_Putchar('G');
#endif
	send_state_packet(1);
	PORTD = 0x04; // green
	break;
	
      case ACTION_STOP:
	//TCCR2A = (1<<COM2A1)|(0<<COM2A0)|(0<<CS22)|(0<<CS21)|(0<<CS20); // stop timer2
	TCCR2A = (0<<COM2A1)|(0<<COM2A0)|(0<<CS22)|(0<<CS21)|(0<<CS20); // stop timer2
#ifdef DEBUGTRIG
	UART_Putchar('S');
#endif
	send_state_packet(0);
	PORTD = 0x01; // red
	break;
	
      case ACTION_GETFREQ:
	send_freq_packet();
	break;      
	
      case ACTION_GETTEMP:
	send_temperature_packet(ADC_read());
	break;      
      
      case ACTION_GETTIME:
	send_timestamp_packet(PT_QUERY_TS,get_time());
	break;      
	
      case ACTION_NONE:
      default:
	break;
	
      }
    }
  }
}
