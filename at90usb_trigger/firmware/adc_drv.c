//! @file $RCSfile: adc_drv.c,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Please read file license.txt for copyright notice.
//!
//! @brief This file contains the low level functions for the ADC
//!
//! @version $Revision: 1.3.2.5 $ $Name: at90usb128-demo-hidgen-last $ $Id: adc_drv.c,v 1.3.2.5 2005/09/05 08:23:26 ebouin Exp $
//!
//! @todo
//! @bug

//_____  I N C L U D E S ___________________________________________________

#include "config.h"

#ifdef USE_ADC //!< this define is set on config.h file
#include "adc_drv.h"
#include <avr/io.h>

//_____ M A C R O S ________________________________________________________


//_____ P R I V A T E    D E C L A R A T I O N _____________________________


//_____ D E F I N I T I O N ________________________________________________


//_____ D E C L A R A T I O N ______________________________________________
//! Configures the ADC accordingly to the ADC Define Configuration values.
//! Take care that you have to select the ports which will be converted as
//! analog inputs thanks to the DIDR0 and DIDR1 registers.
//!
void init_adc(void)
{
    Enable_adc();
#   if (ADC_RIGHT_ADJUST_RESULT == 1)
       Right_adjust_adc_result();
#   elif (ADC_RIGHT_ADJUST_RESULT == 0)
       Left_adjust_adc_result();
#   else
#      error (ADC_RIGHT_ADJUST_RESULT should be 0 or 1... See config.h file)
#   endif

       //#   if (ADC_HIGH_SPEED_MODE == 1)
       //       Enable_adc_high_speed_mode();
       //#   elif (ADC_HIGH_SPEED_MODE == 0)
       //       Disable_adc_high_speed_mode();
       //#   else
       //#      error (ADC_HIGH_SPEED_MODE should be 0 or 1... See config.h file)
       //#   endif

#   if (ADC_INTERNAL_VREF == 2)
       Enable_vcc_vref();
#   elif (ADC_INTERNAL_VREF == 1)
       Enable_internal_vref();
#   elif (ADC_INTERNAL_VREF == 0)
       Enable_vcc_vref();
#   else
#      error (ADC_INTERNAL_VREF should be 0, 1 or 2... See config.h file)
#   endif

#   if (ADC_IT == 1)
       Enable_all_it();
       Enable_adc_it();
#   elif (ADC_IT == 0)
       Disable_adc_it();
#   else
#      error (ADC_IT should be 0 or 1... See config.h file)
#   endif

#   if (ADC_PRESCALER == 128)
       Set_prescaler(128);
#   elif (ADC_PRESCALER == 64)
       Set_prescaler(64);
#   elif (ADC_PRESCALER == 32)
       Set_prescaler(32);
#   elif (ADC_PRESCALER == 16)
       Set_prescaler(16);
#   elif (ADC_PRESCALER ==  8)
       Set_prescaler( 8);
#   elif (ADC_PRESCALER ==  4)
       Set_prescaler( 4);
#   elif (ADC_PRESCALER ==  2)
       Set_prescaler( 0);
#   else
#      error (ADC_PRESCALER should be 2, 4, 8, 16, 32, 64 or 128... See config.h file)
#   endif
}

#endif // USE_ADC
