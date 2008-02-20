/**
 * @file $RCSfile: usb_task.h,v $
 *
 * Copyright (c) 2004 Atmel.
 *
 * Please read file license.txt for copyright notice.
 *
 * @brief This file contains the function declarations
 *
 * @version $Revision: 1.2 $ $Name: at90usb128-demo-hidgen-last $ $Id: usb_task.h,v 1.2 2005/10/11 15:23:55 rletendu Exp $
 *
 * @todo
 * @bug
 */

#ifndef _USB_TASK_H_
#define _USB_TASK_H_

//_____ I N C L U D E S ____________________________________________________


//_____ M A C R O S ________________________________________________________

#define EVT_USB                        0x60               // USB Event
#define EVT_USB_POWERED               (EVT_USB+1)         // USB plugged
#define EVT_USB_UNPOWERED             (EVT_USB+2)         // USB un-plugged
#define EVT_USB_DEVICE_FUNCTION       (EVT_USB+3)         // USB in device
#define EVT_USB_HOST_FUNCTION         (EVT_USB+4)         // USB in host
#define EVT_USB_SUSPEND               (EVT_USB+5)         // USB suspend
#define EVT_USB_WAKE_UP               (EVT_USB+6)         // USB wake up
#define EVT_USB_RESUME                (EVT_USB+7)         // USB resume
#define EVT_USB_RESET                 (EVT_USB+8)         // USB hight speed

//_____ D E C L A R A T I O N S ____________________________________________

void usb_task_init     (void);
void usb_start_device  (void);
void usb_start_host    (void);
void usb_task          (void);


#endif /* _USB_TASK_H_ */

