//! @file $RCSfile: usb_specific_request.c,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Use of this program is subject to Atmel's End User License Agreement.
//! Please read file license.txt for copyright notice.
//!
//! @brief user call-back functions
//!
//! This file contains the user call-back functions corresponding to the
//! application:
//! MASS STORAGE DEVICE
//!
//! @version $Revision: 1.4.6.1 $ $Name: at90usb128-demo-hidgen-last $ $Id: usb_specific_request.c,v 1.4.6.1 2005/11/16 18:26:23 rletendu Exp $
//!
//! @todo
//! @bug


//_____ I N C L U D E S ____________________________________________________

#include "config.h"
#include "conf_usb.h"
#include "usb_drv.h"
#include "usb_descriptors.h"
#include "usb_standard_request.h"
#include "usb_specific_request.h"
#include <avr/pgmspace.h>

//_____ M A C R O S ________________________________________________________

//_____ D E F I N I T I O N ________________________________________________

//_____ P R I V A T E   D E C L A R A T I O N ______________________________

extern uint8_t *pbuffer;
extern uint8_t   data_to_transfer;
extern uint16_t  wInterface;


//_____ D E C L A R A T I O N ______________________________________________

//! usb_user_read_request(type, request);
//!
//! This function is called by the standard usb read request function when
//! the Usb request is not supported. This function returns TRUE when the
//! request is processed. This function returns FALSE if the request is not
//! supported. In this case, a STALL handshake will be automatically
//! sent by the standard usb read request function.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none
//!
Bool usb_user_read_request(uint8_t type, uint8_t request)
{
uint8_t  descriptor_type ;
uint8_t  string_type     ;

   string_type     = Usb_read_byte();
	descriptor_type = Usb_read_byte();
	switch(request)
	{
		case GET_DESCRIPTOR:

			switch (descriptor_type)
			{
			   default:
					return FALSE;
      			break;
			}
			break;
		case SET_CONFIGURATION:
			switch (descriptor_type)
			{
			   default:
					return FALSE;
      			break;
			}
			break;
   case GET_INTERFACE:
//      usb_hid_set_idle();
      usb_hid_get_interface();
      return TRUE;
      break;

 		default:
			return FALSE;
			break;

	}
  	return FALSE;
}



//! usb_user_endpoint_init.
//!
//! This function configures the endpoints.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none
//!
void usb_user_endpoint_init(uint8_t conf_nb)
{
  usb_configure_endpoint(ENDPOINT_BULK_OUT,      \
                         TYPE_BULK,     \
                         DIRECTION_OUT,  \
                         EP_1_MAX_LENGTH_CODE, \
                         ONE_BANK,     \
                         NYET_ENABLED);
  usb_configure_endpoint(ENDPOINT_BULK_IN,      \
                         TYPE_BULK,     \
                         DIRECTION_IN,  \
                         EP_2_MAX_LENGTH_CODE, \
                         ONE_BANK,     \
                         NYET_ENABLED);
}


//! usb_user_get_descriptor.
//!
//! This function returns the size and the pointer on a user information
//! structure
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none
//!
Bool usb_user_get_descriptor(uint8_t type, uint8_t string)
{
	switch(type)
	{
		case STRING_DESCRIPTOR:
      	switch (string)
      	{
        		case LANG_ID:
          		data_to_transfer = sizeof (usb_user_language_id);
          		pbuffer = &(usb_user_language_id.bLength);
					return TRUE;
          		break;
        		case MAN_INDEX:
         	 	data_to_transfer = sizeof (usb_user_manufacturer_string_descriptor);
         	 	pbuffer = &(usb_user_manufacturer_string_descriptor.bLength);
					return TRUE;
          		break;
        		case PROD_INDEX:
         		data_to_transfer = sizeof (usb_user_product_string_descriptor);
          		pbuffer = &(usb_user_product_string_descriptor.bLength);
					return TRUE;
          		break;
        		case SN_INDEX:
          		data_to_transfer = sizeof (usb_user_serial_number);
          		pbuffer = &(usb_user_serial_number.bLength);
					return TRUE;
          		break;
        		default:
          		return FALSE;
			}
		default:
			return FALSE;
	}

	return FALSE;
}



//! usb_hid_set_idle.
//!
//! This function manages hid set idle request.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none
//!
void usb_hid_set_idle (void)
{
  uint8_t dummy;
  dummy = Usb_read_byte();
  dummy = Usb_read_byte();
  wInterface=Usb_read_byte();
  wInterface+=(Usb_read_byte()<<8);

  Usb_ack_receive_setup();

  Usb_send_control_in();                       /* send a ZLP for STATUS phase */
  while(!Is_usb_in_ready());
}


//! usb_hid_get_interface.
//!
//! This function manages hid get interface request.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none
//!
void usb_hid_get_interface (void)
{
  uint8_t dummy;
  dummy = Usb_read_byte();
  dummy = Usb_read_byte();
  wInterface=Usb_read_byte();
  wInterface+=(Usb_read_byte()<<8);

  Usb_ack_receive_setup();

  Usb_send_control_in();                       /* send a ZLP for STATUS phase */
  while(!Is_usb_in_ready());
}

//! hid_get_hid_descriptor.
//!
//! This function manages hid get hid descriptor request.
//!
//! @warning Code:xx bytes (function code length)
//!
//! @param none
//!
//! @return none
//!
void hid_get_hid_descriptor(void)
{

uint16_t wLength;
uint8_t  nb_byte;
bit zlp;



   wInterface=Usb_read_byte();
   wInterface+=(Usb_read_byte()<<8);

   data_to_transfer = 0;
   //data_to_transfer = sizeof(usb_conf_desc.hid);
   //pbuffer = &(usb_conf_desc.hid.bLength);

   wLength = Usb_read_byte();      //!< read wLength
   wLength+= (Usb_read_byte()<<8);
   Usb_ack_receive_setup() ;                  //!< clear the receive setup flag

   if (wLength > data_to_transfer)
   {
      if ((data_to_transfer % EP_CONTROL_LENGTH) == 0) { zlp = TRUE; }
      else { zlp = FALSE; }                   //!< no need of zero length packet
   }
   else
   {
      data_to_transfer = (uint8_t)wLength;         //!< send only requested number of data
   }

   while((data_to_transfer != 0) && (!Is_usb_receive_out()))
   {
      while(!Is_usb_read_control_enabled());

		nb_byte=0;
      while(data_to_transfer != 0)			//!< Send data until necessary
      {
			if(nb_byte++==EP_CONTROL_LENGTH) //!< Check endpoint 0 size
			{
				break;
			}
         Usb_write_byte(pgm_read_byte(pbuffer));
         pbuffer ++;
         data_to_transfer --;
      }
      Usb_send_control_in();
   }

   Usb_send_control_in();

   if(Is_usb_receive_out()) { Usb_ack_receive_out(); return; } //!< abort from Host
   if(zlp == TRUE)        { Usb_send_control_in(); }

   while(!Is_usb_receive_out());
   Usb_ack_receive_out();
}
