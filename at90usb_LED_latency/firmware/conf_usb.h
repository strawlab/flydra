//! @file $RCSfile: conf_usb.h,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Please read file license.txt for copyright notice.
//!
//! This file contains the possible external configuration of the USB
//! This file will be given to any external customer
//!
//! @version $Revision: 1.1 $ $Name: at90usb128-demo-hidgen-last $ $Id: conf_usb.h,v 1.1 2005/11/16 18:25:09 rletendu Exp $
//!
//! @todo
//! @bug
#ifndef _CONF_USB_H_
#define _CONF_USB_H_


#define Usb_unicode(a)			((uint16_t)(a))


//_____ U S B    D E S C R I P T O R    T A B L E S ________________________


#define NB_ENDPOINTS          2  // number of endpoints in the application
#define EP_HID_IN             1
#define EP_HID_OUT            2


#define ENDPOINT_0            0x00  //  OUT EP
#define ENDPOINT_1            0x81  //  IN EP
#define ENDPOINT_2            0x02  //  OUT EP



    // write here the action to associate to each USB event
    // be carefull not to waste time in order not disturbing the functions

#define Usb_sof_action()        sof_action()
#define Usb_wake_up_action()
#define Usb_resume_action()
#define Usb_suspend_action()
#define Usb_reset_action()
#define Usb_vbus_on_action()
#define Usb_vbus_off_action()
#define Usb_set_configuration_action()

extern void sof_action(void);

#endif  //! _CONF_USB_H_


