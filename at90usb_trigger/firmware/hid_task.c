// leds 1 and 2
#define DEBUG_PWM
#define LARGE_BUFFER

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

void set_OCR3C(uint16_t val) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint8_t sreg;

  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  OCR3C = val;
  SREG = sreg; // restore arithmetic state 

  // presumably interrupts are enabled upon function return
}

void set_ICR3(uint16_t val) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet
  // icr1 is TOP for timer1

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

uint16_t get_TCNT3(void) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint16_t val;
  uint8_t sreg;
  
  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  val = TCNT3;
  SREG = sreg; // restore arithmetic state 

  return val;
}

uint16_t get_ICR3(void) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet
  // icr1 is TOP for timer1

  uint16_t val;
  uint8_t sreg;
  
  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  val = ICR3;
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

uint16_t get_OCR3B(void) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint16_t val;
  uint8_t sreg;
  
  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  val = OCR3B;
  SREG = sreg; // restore arithmetic state 

  return val;
}

uint16_t get_OCR3C(void) {
  // See "Accessing 16-bit Registers" of the AT90USB1287 datasheet

  uint16_t val;
  uint8_t sreg;
  
  sreg = SREG; // save arithmetic state
  cli(); // disable interrupts
  val = OCR3C;
  SREG = sreg; // restore arithmetic state 

  return val;
}

void init_pwm_output(void) {
  /*
    
  n = 3 (timer3)

  Set frequency of PWM using ICRn to set TOP. (Not double-buffered,
  also, clear TCNT before setting.)  
  
  Set compare value using OCRnA.
  
  WGMn3:0 = 14
  
  */

  // set output direction on pin
  PORTC &= 0x8F; // pin C4-6 set low to start
  DDRC |= 0x70; // enable output for Output compare and PWM A-C of Timer/Counter 3

  // Set output compare to mid-point
  set_OCR3A( 0x7FFF );

  set_OCR3B( 0x0 );
  set_OCR3C( 0x0 );

  // Set TOP high
  set_ICR3( 0xFFFF );

  // ---- set TCCR1A ----------
  // set Compare Output Mode for Fast PWM
  // COM3A1:0 = 1,0 clear OC3A on compare match
  // COM3B1:0 = 1,0 clear OC3B on compare match
  // COM3C1:0 = 1,0 clear OC3B on compare match
  // WGM31, WGM30 = 1,0
  TCCR3A = 0xAA;

  // ---- set TCCR1B ----------
  // high bits = 0,0,0
  //WGM33, WGM32 = 1,1
  // CS1 = 0,0,1 (starts timer1) (clock select)
  TCCR3B = 0x19;

#ifdef DEBUG_PWM
  TIMSK3 = 0x07; //OCIE1A|TOIE1; // enable interrucpts
#endif
}

#ifdef DEBUG_PWM
ISR(TIMER3_COMPA_vect) {
  Led1_off();
}
ISR(TIMER3_COMPB_vect) {
  Led2_off();
}
ISR(TIMER3_OVF_vect) {
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
   
   init_pwm_output(); // trigger output
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
#define TASK_FLAGS_NEW_TIMER3_DATA 0x02

   uint8_t clock_select_timer3=0;
   uint32_t volatile tmp;

   uint16_t new_ocr3a;
   uint16_t new_ocr3b;
   uint16_t new_ocr3c;
   uint16_t new_icr3; // icr3 is TOP for timer3

   if(usb_connected)                     //! Check USB HID is enumerated
    {
      Usb_select_endpoint(ENDPOINT_BULK_OUT);    //! Get Data from Host
      if(Is_usb_receive_out()) {
	// first 8 bytes
	new_ocr3a =           Usb_read_byte()<<8; // high byte
	new_ocr3a +=          Usb_read_byte();    // low byte
	new_ocr3b =           Usb_read_byte()<<8; // high byte
	new_ocr3b +=          Usb_read_byte();    // low byte

#ifdef LARGE_BUFFER
	new_ocr3c =           Usb_read_byte()<<8; // high byte
	new_ocr3c +=          Usb_read_byte();    // low byte
#else
	new_ocr3c =           0;
#endif
	new_icr3  =           Usb_read_byte()<<8; // high byte  // icr3 is TOP for timer3
	new_icr3 +=           Usb_read_byte();    // low byte

	// next 8 bytes
	flags     =           Usb_read_byte();
	clock_select_timer3 = Usb_read_byte();
#ifdef LARGE_BUFFER
	Usb_read_byte();
	Usb_read_byte();

	Usb_read_byte();
	Usb_read_byte();
	Usb_read_byte();
	Usb_read_byte();
#endif
	Usb_ack_receive_out();

	if (flags & TASK_FLAGS_NEW_TIMER3_DATA) {
	  // update timer3
	  set_OCR3A(new_ocr3a);
	  set_OCR3B(new_ocr3b);
	  set_OCR3C(new_ocr3c);
	  set_ICR3(new_icr3);  // icr3 is TOP for timer3
	  TCCR3B = (TCCR3B & 0xF8) | (clock_select_timer3 & 0x07); // low 3 bits sets CS
	  new_data = TRUE;
	}

      }
      if (flags & TASK_FLAGS_ENTER_DFU) //! Check if we received DFU mode command from host
	{
	  Usb_detach();                    // detach from USB...
	  TCCR3B = 0x00; // disable trigger outputs and timer3
	  Led1_off();
	  Led2_off();
	  Led3_off();
	  Led0_on();
	  for(tmp=0;tmp<70000;tmp++);     // pause...
	  Led3_on();
	  (*start_bootloader)();
	}

      if (new_data == TRUE) {

	Usb_select_endpoint(ENDPOINT_BULK_IN);    //! Ready to send these information to the host application
	if(Is_usb_in_ready())
	  {
	    new_ocr3a = get_OCR3A();
	    new_ocr3b = get_OCR3B();
	    new_ocr3c = get_OCR3C();
	    new_icr3 = get_ICR3();  // icr1 is TOP for timer1
	    
	    Usb_write_byte((uint8_t)(new_ocr3a>>8));
	    Usb_write_byte((uint8_t)(new_ocr3a));
	    Usb_write_byte((uint8_t)(new_ocr3b>>8));
	    Usb_write_byte((uint8_t)(new_ocr3b));

	    Usb_write_byte((uint8_t)(new_ocr3c>>8));
	    Usb_write_byte((uint8_t)(new_ocr3c));
	    Usb_write_byte((uint8_t)(new_icr3>>8));
	    Usb_write_byte((uint8_t)(new_icr3));

	    Usb_write_byte(PORTC);
	    Usb_write_byte(TCCR3B);
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
