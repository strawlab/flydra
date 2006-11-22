/*H**************************************************************************
* $RCSfile: usb_specific_request.h,v $
*----------------------------------------------------------------------------
* Copyright (c) 2004 Atmel.
*----------------------------------------------------------------------------
* RELEASE:      $Name: at90usb128-demo-hidgen-last $
* REVISION:     $Revision: 1.3.6.1 $
* FILE_CVSID:   $Id: usb_specific_request.h,v 1.3.6.1 2005/11/16 18:26:23 rletendu Exp $
*----------------------------------------------------------------------------
* PURPOSE:
* This file contains the user call-back functions corresponding to the
* application:
* MASS STORAGE DEVICE
*****************************************************************************/

#ifndef _USB_USER_ENUM_H_
#define _USB_USER_ENUM_H_

/*_____ I N C L U D E S ____________________________________________________*/

#include "config.h"
#include <avr/pgmspace.h>

/*_____ M A C R O S ________________________________________________________*/

extern const S_usb_device_descriptor PROGMEM usb_dev_desc;
extern const S_usb_user_configuration_descriptor PROGMEM usb_conf_desc;
extern const S_usb_manufacturer_string_descriptor PROGMEM usb_user_manufacturer_string_descriptor;
extern const S_usb_product_string_descriptor PROGMEM usb_user_product_string_descriptor;
extern const S_usb_serial_number PROGMEM usb_user_serial_number;
extern const S_usb_language_id PROGMEM usb_user_language_id;



/*_____ D E F I N I T I O N ________________________________________________*/
Bool  usb_user_read_request(uint8_t, uint8_t);
Bool  usb_user_get_descriptor(uint8_t , uint8_t);
void  usb_user_endpoint_init(uint8_t);
void hid_get_report(void);
void hid_set_report(void);
void usb_hid_get_interface(void);
void usb_hid_set_idle(void);
void hid_get_hid_descriptor(void);

#endif

