// leds 1 and 2
#define DEBUG_PWM
// led 4
#define DEBUG_LASER

//_____  I N C L U D E S ___________________________________________________

#include "config.h"
#include "conf_usb.h"
#include "hid_task.h"
#include "stk_525.h"
#include "usb_drv.h"
#include "usb_descriptors.h"
#include "usb_standard_request.h"
#include "usb_specific_request.h"
#include "adc_drv.h"



//_____ M A C R O S ________________________________________________________


//_____ D E F I N I T I O N S ______________________________________________



//_____ D E C L A R A T I O N S ____________________________________________

extern bit   usb_connected;
bit   new_data=FALSE;
extern  uint8_t   usb_configuration_nb;
volatile uint8_t cpt_sof=0;

//! Declare function pointer to USB bootloader entry point
void (*start_bootloader) (void)=(void (*)(void))0xf000;


void set_OCR1A(uint16_t val) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint8_t sreg;

  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  OCR1A = val;
  SREG = sreg; // restore arithmetic state 

  // presumably interrupts are enabled upon function return
}

void set_OCR1B(uint16_t val) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint8_t sreg;

  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  OCR1B = val;
  SREG = sreg; // restore arithmetic state 

  // presumably interrupts are enabled upon function return
}

void set_OCR3A(uint16_t val) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint8_t sreg;

  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  OCR3A = val;
  SREG = sreg; // restore arithmetic state 

  // presumably interrupts are enabled upon function return
}

void set_OCR3B(uint16_t val) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint8_t sreg;

  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  OCR3B = val;
  SREG = sreg; // restore arithmetic state 

  // presumably interrupts are enabled upon function return
}

void set_ICR1(uint16_t val) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet
  // icr1 is TOP for timer1

  uint8_t sreg;

  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  ICR1 = val;
  SREG = sreg; // restore arithmetic state 

  // presumably interrupts are enabled upon function return
}

void set_ICR3(uint16_t val) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet
  uint8_t sreg;

  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  ICR3 = val;
  SREG = sreg; // restore arithmetic state 

  // presumably interrupts are enabled upon function return
}

void set_TCNT3(uint16_t val) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet
  uint8_t sreg;

  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  TCNT3 = val;
  SREG = sreg; // restore arithmetic state 

  // presumably interrupts are enabled upon function return
}

uint16_t get_TCNT1(void) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint16_t val;
  uint8_t sreg;
  
  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  val = TCNT1;
  SREG = sreg; // restore arithmetic state 

  return val;
}

uint16_t get_ICR1(void) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet
  // icr1 is TOP for timer1

  uint16_t val;
  uint8_t sreg;
  
  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  val = ICR1;
  SREG = sreg; // restore arithmetic state 

  return val;
}

uint16_t get_OCR1A(void) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint16_t val;
  uint8_t sreg;
  
  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  val = OCR1A;
  SREG = sreg; // restore arithmetic state 

  return val;
}

uint16_t get_OCR1B(void) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint16_t val;
  uint8_t sreg;
  
  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  val = OCR1B;
  SREG = sreg; // restore arithmetic state 

  return val;
}

uint16_t get_OCR3A(void) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint16_t val;
  uint8_t sreg;
  
  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  val = OCR3A;
  SREG = sreg; // restore arithmetic state 

  return val;
}

void init_pwm_output(void) {
  /*
    
  n = 1 (timer1)

  Set frequency of PWM using ICRn to set TOP. (Not double-buffered,
  also, clear TCNT before setting.)  
  
  Set compare value using OCRnA.
  
  WGMn3:0 = 14
  
  */

  // set output direction on pin
  PORTB &= 0x9F; // pin B5,B6 is set low to start
  DDRB |= 0x60; // enable output for Output compare and PWM A and B of Timer/Counter 1

  // Set output compare to mid-point
  set_OCR1A( 0x7FFF );
  set_OCR1B( 0x7FFF );

  // Set TOP high
  set_ICR1( 0xFFFF );

  // ---- set TCCR1A ----------
  // set Compare Output Mode for Fast PWM
  // COM1A1:0 = 1,0 clear OC1A on compare match
  // COM1B1:0 = 1,0 clear OC1B on compare match
  // COM1C1:0 = 0,0 OCR1C disconnected
  // WGM11, WGM10 = 1,0
  TCCR1A = 0xA2;

  // ---- set TCCR1B ----------
  // high bits = 0,0,0
  //WGM13, WGM12 = 1,1
  // CS1 = 0,0,1 (starts timer1) (clock select)
  TCCR1B = 0x19;

#ifdef DEBUG_PWM
  TIMSK1 = 0x07; //OCIE1A|TOIE1; // enable interrucpts
#endif
}

void init_cam_laser_trig_output(void) {
  uint16_t TOP=40816;
  uint16_t pulse_dur=500;
  /*
    
  n = 3 (timer3)

  Set frequency of PWM using ICRn to set TOP. (Not double-buffered,
  also, clear TCNT before setting.)  
  
  Set compare value using OCRnA.
  
  WGMn3:0 = 14
  
  */

  // set output direction on pin
  PORTC &= 0x9F; // pin C6, C5 is set low to start
  DDRC |= 0x60; // enable output for Output compare and PWM A,B of Timer/Counter 3
  //DDRC = 0; // enable output for Output compare and PWM A,B of Timer/Counter 3

  // Set output compare to mid-point
  set_OCR3A( 1000 );
  //set_OCR3B( 0x7FFF );
  //set_OCR3B( 0x4000 ); // longer pulse
  //set_OCR3B( 1638 ); // short pulse
  set_OCR3B( TOP-pulse_dur ); // short pulse

  // Set TOP high
  // TOP = FOSC / CLOCK_SELECT / CAM_FRAMERATE
  //set_ICR3( 0xa2c2 ); // 24 Hz @ 8000000 Hz (FOSC) / 8 (clock select divide by 8)
  set_ICR3( TOP ); // 24.5 Hz @ 8000000 Hz (FOSC) / 8 (clock select divide by 8)

  // ---- set TCCR3A ----------
  // set Compare Output Mode for Fast PWM
  // COM3A1:0 = 1,1 set OC3A on compare match, clear on TOP // laser
  // COM3B1:0 = 1,0 clear OC3B on compare match, set on TOP // camera
  // COM3C1:0 = 0,0 OCR3C disconnected
  // WGM31, WGM30 = 1,0
  //TCCR3A = 0xE2;
  //TCCR3A = 0xB2; // stuff above is wrong. 1011, want to flip camera

  // COM3A1:0 = 1,1 set OC3A on compare match, clear on TOP // laser
  // COM3B1:0 = 1,0 set OC3B on compare match, clear on TOP // camera
  TCCR3A = 0xF2; // stuff above is wrong. 1111, want to flip camera

  // ---- set TCCR3B ----------
  // high bits = 0,0,0
  //WGM33, WGM32 = 1,1
  // CS1 = 0,1,0 (starts timer3) (clock select = 8)
  TCCR3B = 0x1A; // orig CS8
  //  TCCR3B = 0x1D; // orig CS8

}

#ifdef DEBUG_PWM
ISR(TIMER1_COMPA_vect) {
  Led1_off();
}
ISR(TIMER1_COMPB_vect) {
  Led2_off();
}
ISR(TIMER1_OVF_vect) {
  Led1_on();
  Led2_on();
}
#endif

//!
//! @brief This function initializes the target board ressources.
//!
//! @warning Code:?? bytes (function code length)
//!
//! @param none
//!
//! @return none
//!
//!/
void hid_task_init(void)
{
   init_adc();
   Leds_init();
   Joy_init();
   
   init_pwm_output(); // servo control
   init_cam_laser_trig_output(); // camera trigger
}





//! @brief Entry point of the HID generic communication task
//!
//! This function manages IN/OUT repport management.
//!
//! @warning Code:?? bytes (function code length)
//!
//! @param none
//!
//! @return none
void hid_task(void)
{
   uint8_t flags=0;
#define TASK_FLAGS_ENTER_DFU 0x01
#define TASK_FLAGS_NEW_TIMER1_DATA 0x02
#define TASK_FLAGS_NEW_TIMER3A_WIDTH 0x04
#define TASK_FLAGS_NEW_TIMER3_ALLDATA 0x08

   uint8_t clock_select_timer1=0;
   uint32_t volatile tempo;

   uint16_t new_ocr1a;
   uint16_t new_ocr1b;
   uint16_t new_icr1; // icr1 is TOP for timer1

   //   uint8_t new_timer3_and_clock_select_compressed=0;
   uint16_t new_ocr3a;
   uint16_t test_ocr3a;
   uint16_t new_ocr3b;
   uint16_t new_icr3; // icr3 is TOP for timer3
   uint8_t new_tccr3a, new_tccr3b;

   if(usb_connected)                     //! Check USB HID is enumerated
    {
      Usb_select_endpoint(ENDPOINT_BULK_OUT);    //! Get Data from Host
      if(Is_usb_receive_out()) {
	// first 8 bytes
	new_ocr1a =           Usb_read_byte()<<8; // high byte
	new_ocr1a +=          Usb_read_byte();    // low byte
	new_ocr1b =           Usb_read_byte()<<8; // high byte
	new_ocr1b +=          Usb_read_byte();    // low byte

	new_icr1  =           Usb_read_byte()<<8; // high byte  // icr1 is TOP for timer1
	new_icr1 +=           Usb_read_byte();    // low byte
	flags     =           Usb_read_byte();
	clock_select_timer1 = Usb_read_byte();

	// next 8 bytes
	new_ocr3a =  Usb_read_byte()<<8; // high byte
 	new_ocr3a += Usb_read_byte();    // low byte
	new_ocr3b =  Usb_read_byte()<<8; // high byte
 	new_ocr3b += Usb_read_byte();    // low byte

	new_icr3 =  Usb_read_byte()<<8; // high byte  // icr3 is TOP for timer3
	new_icr3 += Usb_read_byte();    // low byte
	new_tccr3a= Usb_read_byte();
	new_tccr3b= Usb_read_byte();

	Usb_ack_receive_out();

	if (flags & TASK_FLAGS_NEW_TIMER1_DATA) {
	  // update timer1
	  set_OCR1A(new_ocr1a);
	  set_OCR1B(new_ocr1b);
	  set_ICR1(new_icr1);  // icr1 is TOP for timer1
	  TCCR1B = (TCCR1B & 0xF8) | (clock_select_timer1 & 0x07); // low 3 bits sets CS
	  new_data = TRUE;
	}

	// check for new timer3 info
	if (flags & TASK_FLAGS_NEW_TIMER3A_WIDTH) {
	  set_OCR3A(new_ocr3a);
	}

	if (flags & TASK_FLAGS_NEW_TIMER3_ALLDATA) {
	  TCCR3B = (new_tccr3b & 0xF8); // stop timer
	  PORTC &= 0xDF;
	  TCCR3A = new_tccr3a;
	  set_ICR3(new_icr3);  // icr1 is TOP for timer1

	  /*
	  if (!(new_tccr3a & 0xa0)) {
	    // output compare a not enabled, make sure value is low
	    PORTC &= 0xDF; // pin C5 is set low
	    //PORTC &= 0x9F; // pins C5,C6 are set low
	  }
	  */

	  set_OCR3A(new_ocr3a);
	  /*
	  test_ocr3a = get_OCR3A();
	  if (test_ocr3a != new_ocr3a) {
	    test_ocr3a = 0;
	  }
	  */
	  set_OCR3B(new_ocr3b);

	  set_TCNT3( new_icr3-20 ); // roll timer over immediately
	  TCCR3B = new_tccr3b; // restart timer
	}

      }
      if (flags & TASK_FLAGS_ENTER_DFU) //! Check if we received DFU mode command from host
	{
	  Usb_detach();                    //! Detach actual generic HID application
	  for(tempo=0;tempo<70000;tempo++);//! Wait some time before
	  (*start_bootloader)();           //! Jumping to booltoader
	}

      if (new_data == TRUE) {

	Usb_select_endpoint(ENDPOINT_BULK_IN);    //! Ready to send these information to the host application
	if(Is_usb_in_ready())
	  {
	    new_ocr1a = get_OCR1A();
	    new_ocr1b = get_OCR1B();
	    new_icr1 = get_ICR1();  // icr1 is TOP for timer1
	    
	    Usb_write_byte((uint8_t)(new_ocr1a>>8));
	    Usb_write_byte((uint8_t)(new_ocr1a));
	    Usb_write_byte((uint8_t)(new_ocr1b>>8));
	    Usb_write_byte((uint8_t)(new_ocr1b));

	    Usb_write_byte((uint8_t)(new_icr1>>8));
	    Usb_write_byte((uint8_t)(new_icr1));
	    Usb_write_byte(PORTB);
	    Usb_write_byte(TCCR1B);

	    Usb_write_byte(0x77);
	    Usb_write_byte(0x77);
	    Usb_write_byte(0x77);
	    Usb_write_byte(0x77);
	    
	    Usb_write_byte(0x77);
	    Usb_write_byte(0x77);
	    Usb_write_byte(0x77);
	    Usb_write_byte(0x77);
	    
	    Usb_ack_fifocon();               //! Send data over the USB
	    new_data = FALSE;
	  }
      }
    }

}

//! @brief sof_action
//!
//! This function increments the cpt_sof counter each times
//! the USB Start Of Frame interrupt subroutine is executed (1ms)
//! Usefull to manage time delays
//!
//! @warning Code:?? bytes (function code length)
//!
//! @param none
//!
//! @return none
void sof_action()
{
   cpt_sof++;
}
