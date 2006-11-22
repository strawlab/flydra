//! @file $RCSfile: config.h,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Please read file license.txt for copyright notice.
//!
//! This file contains the system configuration definition
//!
//! @version $Revision: 1.1 $ $Name: at90usb128-demo-hidgen-last $ $Id: config.h,v 1.1 2005/11/16 18:25:10 rletendu Exp $
//!
//! @todo
//! @bug

#ifndef _CONFIG_H_
#define _CONFIG_H_


//_____ I N C L U D E S ____________________________________________________
#include <avr/io.h>
#include <avr/interrupt.h>

#include "conf_scheduler.h"

#define FOSC            8000       // Oscillator frequency(KHz), also see F_CPU in Makefile
//****** END Generic Configuration ******

#define USE_ADC
#define ADC_PRESCALER 64
#define ADC_RIGHT_ADJUST_RESULT 1
#define ADC_INTERNAL_VREF  2     //AVCC As reference voltage

#define Bool uint8_t
#define bit uint8_t
#define FALSE 0
#define TRUE 1

#endif  //! _CONFIG_H_

