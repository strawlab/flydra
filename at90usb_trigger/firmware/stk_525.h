//! @file $RCSfile: stk_525.h,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Please read file license.txt for copyright notice.
//!
//! @brief This file contains the low level macros and definition for stk525 board
//!
//! @version $Revision: 1.7 $ $Name: at90usb128-demo-hidgen-last $
//!
//! @todo
//! @bug

#ifndef STK_525_H
#define STK_525_H

//_____ I N C L U D E S ____________________________________________________
#include "config.h"


//_____ M A C R O S ________________________________________________________

//! @defgroup STK525_module STK525 Module
//! STK525 Module
//! @{


      //! @defgroup STK 525 Leds Management
      //! Macros to manage Leds on STK525
      //! @{
#define Leds_init()     (DDRD |= 0xF0)
#define Leds_on()       (PORTD |= 0xF0)
#define Leds_off()      (PORTD &= 0x0F)
#define Led0_on()       (PORTD |= 0x10)
#define Led1_on()       (PORTD |= 0x20)
#define Led2_on()       (PORTD |= 0x40)
#define Led3_on()       (PORTD |= 0x80)
#define Led0_off()      (PORTD &= 0xEF)
#define Led1_off()      (PORTD &= 0xDF)
#define Led2_off()      (PORTD &= 0xBF)
#define Led3_off()      (PORTD &= 0x7F)
#define Led0_toggle()   (PIND |= 0x10)
#define Led1_toggle()   (PIND |= 0x20)
#define Led2_toggle()   (PIND |= 0x40)
#define Led3_toggle()   (PIND |= 0x80)
#define Leds_set_val(c) (Leds_off(),PORTD |= (c<<4)&0xF0)
#define Leds_get_val()  (PORTD>>4)
      //! @}

      //! @defgroup STK 525 Joystick Management
      //! Macros to manage Joystick on STK525
      //! @{
#define Joy_init()      (DDRB &= 0x1F, PORTB |= 0xE0, DDRE &= 0xE7, PORTE |= 0x30)
#define Is_joy_up()     ((PINB & 0x80) ?  FALSE : TRUE)
#define Is_joy_left()   ((PINB & 0x40) ?  FALSE : TRUE)
#define Is_joy_select() ((PINB & 0x20) ?  FALSE : TRUE)
#define Is_joy_right()  ((PINE & 0x10) ?  FALSE : TRUE)
#define Is_joy_down()   ((PINE & 0x20) ?  FALSE : TRUE)
      //! @}

      //! @defgroup STK 525 HWB button management
      //! HWB button is connected to PE2 and can also
      //! be used as generic push button
      //! @{
#define Hwb_button_init()      (DDRE &= 0xFB, PORTE |= 0x04)
#define Is_hwb()               ((PINE & 0x04) ?  FALSE : TRUE)
      //! @}

//!< STK 525 ADC Channel Definition
#define ADC_POT_CH   0x01
#define ADC_MIC_CH   0x02
#define ADC_TEMP_CH  0x00

#ifdef USE_ADC       //!< this define is set in config.h file

//! Get_adc_mic_val.
//!
//! This function performs an ADC conversion from the stk525 MIC channel
//! an returns the 10 bits result in an U16 value.
//!
//! @warning USE_ADC should be defined in config.h
//!
//! @param none
//!
//! @return U16 microphone sample value.
//!
   uint16_t Get_adc_mic_val(void);

//! Get_adc_temp_val.
//!
//! This function performs an ADC conversion from the stk525 TEMP channel
//! an returns the 10 bits result in an U16 value.
//!
//! @warning USE_ADC should be defined in config.h
//!
//! @param none
//!
//! @return U16 analog sensor temperature value.
//!
   uint16_t Get_adc_temp_val(void);

//! Get_adc_pot_val.
//!
//! This function performs an ADC conversion from the stk525 POT channel
//! an returns the 10 bits result in an uint16_t value.
//!
//! @warning USE_ADC should be defined in config.h
//!
//! @param none
//!
//! @return uint16_t analog potentiometer value.
//!
   uint16_t Get_adc_pot_val(void);

//! Read_temperature.
//!
//! This function performs an ADC conversion from the stk525 POT channel
//! an returns the 10 bits result of the temperature (in °C) in an int16_t value.
//!
//! @warning USE_ADC should be defined in config.h
//!
//! @param none
//!
//! @return int16_t temperature value in °C.
//!
   int16_t  Read_temperature(void);

#endif  //!USE_ADC

      //! @defgroup TK 525 ATMEL Hardware data flash configuration
      //! Macros to init the environnement for DF on STK525
      //! @{
#define Init_df_stk525()   (DDRB_Bit4=1,PORTB_Bit3=1,PORTB_Bit4=1,PORTB_Bit0=1)
#define DF_CS_PIN PORTB_Bit4
#define DF_CS              DF_CS0
#define DF_CS0             DF_CS_PIN
#define DF_CS1             DF_CS0
#define DF_CS2             DF_CS0
#define DF_CS3             DF_CS0
#define DF_DESEL_ALL       ((Byte)0xF0)        /* set CS# dataflash memories */
#define DF_NB_MEM          1
#define df_init_spi()      Init_df_stk525()
      //! @}

//! @}

#endif  // STK_525_H
