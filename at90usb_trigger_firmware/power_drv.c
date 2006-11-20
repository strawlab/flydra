//! @file $RCSfile: power_drv.c,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Use of this program is subject to Atmel's End User License Agreement.
//! Please read file license.txt for copyright notice.
//!
//! @brief This file contains the Power management driver routines.
//!
//! This file contains the Power management driver routines.
//!
//! @version $Revision: 1.2 $ $Name: at90usb128-demo-hidgen-last $ $Id: power_drv.c,v 1.2 2005/09/30 15:15:03 rletendu Exp $
//!
//! @todo Implemets adc, stanby, extended stanby and power save modes
//! @bug

//_____ I N C L U D E S ____________________________________________________

#include "config.h"
#include "power_drv.h"

//_____ M A C R O S ________________________________________________________

//_____ D E C L A R A T I O N ______________________________________________

//! set_power_down_mode.
//!
//! This function makes the AVR core enter power down mode.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none.
//!
void set_power_down_mode(void)
{
	Setup_power_down_mode();
	Sleep_instruction();
}


//! set_idle_mode.
//!
//! This function makes the AVR core enter idle mode.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none.
//!
void set_idle_mode(void)
{
	Setup_idle_mode();
	Sleep_instruction();
}

//! set_adc_noise_reduction_mode.
//!
//! This function makes the AVR core enter adc noise reduction mode.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none.
//!
void set_adc_noise_reduction_mode(void)
{
	Setup_adc_noise_reduction_mode();
	Sleep_instruction();
}

//! set_power_save_mode.
//!
//! This function makes the AVR core enter power save mode.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none.
//!
void set_power_save_mode(void)
{
	Setup_power_save_mode();
	Sleep_instruction();
}

//! set_standby_mode.
//!
//! This function makes the AVR core enter standby mode.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none.
//!
void set_standby_mode(void)
{
	Setup_standby_mode();
	Sleep_instruction();
}

//! set_ext_standby_mode.
//!
//! This function makes the AVR core enter extended standby mode.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none.
//!
void set_ext_standby_mode(void)
{
	Setup_ext_standby_mode();
	Sleep_instruction();
}
