//!
//! @file $RCSfile: usb_task.c,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Please read file license.txt for copyright notice.
//!
//! @brief This file manages the USB controller.
//!
//! The USB task checks the income of new requests from the USB Host.
//! When a Setup request occurs, this task will launch the processing
//! of this setup contained in the usb_enum.c file.
//! Other class specific requests are also processed in this file.
//! This file manages all the USB events:
//! Suspend / Resume / Reset / Start Of Frame / Wake Up / Vbus events
//!
//! @version $Revision: 1.4 $ $Name: at90usb128-demo-hidgen-last $ $Id: usb_task.c,v 1.4 2005/11/08 15:31:23 rletendu Exp $
//!
//! @todo
//! @bug
//!/

//_____  I N C L U D E S ___________________________________________________

#include "config.h"
#include "conf_usb.h"
#include "usb_task.h"

#include "usb_drv.h"
#include "usb_descriptors.h"
#include "usb_standard_request.h"

#include "power_drv.h"
#include "pll_drv.h"
#include "wdt_drv.h"

//_____ M A C R O S ________________________________________________________

//_____ D E F I N I T I O N S ______________________________________________

//!
//! Public : (bit) usb_connected
//! usb_connected is set to TRUE when VBUS has been detected
//! usb_connected is set to FALSE otherwise
//!/
bit   usb_connected;


static uint8_t g_usb_event;
static bit is_new_usb_event;
extern uint8_t  usb_configuration_nb;
static uint16_t index;

bit reset_detected=FALSE;



#define Usb_send_event(x)               (g_usb_event = x, is_new_usb_event = TRUE)

//_____ D E C L A R A T I O N S ____________________________________________
//!
//! @brief This function initializes the USB the associated variables.
//!
//! This function enables the USB controller and init the USB interrupts.
//! The aim is to allow the USB connection detection in order to send
//! the appropriate USB event to the operating mode manager.
//!
//! @warning Code:?? bytes (function code length)
//!
//! @param none
//!
//! @return none
//!
//!/
void usb_task_init(void)
{
   is_new_usb_event = FALSE;
   sei();
   Usb_force_device_mode();
   Usb_enable();
   Usb_select_device();
	Usb_enable_vbus_interrupt();
   index=0;
}

//!
//! @brief This function initializes the USB device controller
//!
//! This function enables the USB controller and init the USB interrupts.
//! The aim is to allow the USB connection detection in order to send
//! the appropriate USB event to the operating mode manager.
//!
//! @warning Code:?? bytes (function code length)
//!
//! @param none
//!
//! @return none
//!
//!/
void usb_start_device (void)
{
   Usb_enable_regulator();
   Pll_start_auto();
   Wait_pll_ready();
   Usb_unfreeze_clock();
   Usb_enable_suspend_interrupt();
   Usb_enable_reset_interrupt();
   usb_init_device();         // configure the USB controller EP0
   Usb_attach();
   usb_connected = FALSE;
}



//! @brief Entry point of the USB mamnagement
//!
//! This function is the entry point of the USB management. Each USB
//! event is checked here in order to launch the appropriate action.
//! If a Setup request occurs on the Default Control Endpoint,
//! the usb_process_request() function is call in the usb_enum.c file
//! If a new USB mass storage Command Block Wrapper (CBW) occurs,
//! this one will be decoded and the SCSI command will be taken in charge
//! by the scsi decoder.
//!
//! @warning Code:?? bytes (function code length)
//!
//! @param none
//!
//! @return none
void usb_task(void)
{
   if (Is_usb_vbus_high()&& usb_connected==FALSE)
   {
      usb_connected = TRUE;
      Usb_vbus_on_action();
      Usb_send_event(EVT_USB_POWERED);
   	Usb_enable_reset_interrupt();
		Usb_attach();
   }

	// USB EVENT MANAGEMENT
   if (is_new_usb_event == TRUE)
   {
   	is_new_usb_event = FALSE;
     	switch (g_usb_event)
     	{
      	case EVT_USB_POWERED:

         	break;
         case EVT_USB_RESET:

         default :
            break;
      }
    }

   if(reset_detected==TRUE)
   {
      Usb_reset_endpoint(0);
      usb_configuration_nb=0;
      reset_detected=FALSE;
   }

    // USB MANAGEMENT
    Usb_select_endpoint(EP_CONTROL);
    if (Is_usb_receive_setup())
    {
      usb_process_request();
    }
}





//! @brief USB interrupt process
//!
//! This function is called each time a USB interrupt occurs.
//! The following USB events are taken in charge:
//! - VBus On / Off
//! - Start Of Frame
//! - Suspend
//! - Wake-Up
//! - Resume
//! - Reset
//! For each event, the user can launch an action by completing
//! the associate define
//!
//! @warning Code:?? bytes (function code length)
//!
//! @param none
//!
//! @return none


//!
//! @brief General interrupt subroutine
//! Check for VBUS and ID pin transitions
//! @return
//!
ISR(USB_GEN_vect)
{
   if (Is_usb_vbus_transition())
   {
      Usb_ack_vbus_transition();
      if (Is_usb_vbus_high())
      {
         usb_connected = TRUE;
         Usb_vbus_on_action();
         Usb_send_event(EVT_USB_POWERED);
			Usb_enable_reset_interrupt();
         usb_start_device();
			Usb_attach();
      }
      else
      {
         Usb_vbus_off_action();
         usb_connected = FALSE;
         Usb_send_event(EVT_USB_UNPOWERED);
      }
   }

   if(Is_usb_id_transition())
   {
      Usb_ack_id_transition();
      if(Is_usb_id_device())
      {
         Usb_send_event(EVT_USB_DEVICE_FUNCTION);
      }
      else
      {
         Usb_send_event(EVT_USB_HOST_FUNCTION);
      }
   }

   if (Is_usb_sof())
   {
      Usb_ack_sof();
      Usb_sof_action();
   }

   if (Is_usb_suspend())
   {
      Usb_ack_suspend();
      Usb_enable_wake_up_interrupt();
      Usb_ack_wake_up();                 // clear wake up to detect next event
      Usb_freeze_clock();
      Usb_send_event(EVT_USB_SUSPEND);
      Usb_suspend_action();
   }

   if (Is_usb_wake_up())
   {
      Usb_unfreeze_clock();
      Usb_ack_wake_up();
      Usb_disable_wake_up_interrupt();
      Usb_wake_up_action();
      Usb_send_event(EVT_USB_WAKE_UP);
   }

   if (Is_usb_resume())
   {
      Usb_disable_wake_up_interrupt();
      Usb_ack_resume();
      Usb_disable_resume_interrupt();
      Usb_resume_action();
      Usb_send_event(EVT_USB_RESUME);
   }

   if (Is_usb_reset())
   {
      Usb_ack_reset();
      usb_init_device();
      Usb_reset_action();
      Usb_send_event(EVT_USB_RESET);
      reset_detected=TRUE;
   }
}


extern void suspend_action(void)
{
  sei();
  Enter_power_down_mode();
}
