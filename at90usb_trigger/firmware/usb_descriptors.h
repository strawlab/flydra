//! @file $RCSfile: usb_descriptors.h,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Use of this program is subject to Atmel's End User License Agreement.
//! Please read file license.txt for copyright notice.
//!
//! @brief HID generic Identifers.
//!
//! This file contains the usb parameters that uniquely identify the
//! application through descriptor tables.
//!
//! @version $Revision: 1.2 $ $Name: at90usb128-demo-hidgen-last $ $Id: usb_descriptors.h,v 1.2 2006/01/09 12:07:19 rletendu Exp $
//!
//! @todo
//! @bug

#ifndef _USB_USERCONFIG_H_
#define _USB_USERCONFIG_H_

//_____ I N C L U D E S ____________________________________________________

#include "config.h"
#include "usb_standard_request.h"
#include "conf_usb.h"

//_____ M A C R O S ________________________________________________________

#define Usb_get_dev_desc_pointer()        (&(usb_dev_desc.bLength))
#define Usb_get_dev_desc_length()         (sizeof (usb_dev_desc))
#define Usb_get_conf_desc_pointer()       (&(usb_conf_desc.cfg.bLength))
#define Usb_get_conf_desc_length()        (sizeof (usb_conf_desc))

//_____ U S B    D E F I N E _______________________________________________

                  // USB Device descriptor
#define USB_SPECIFICATION     0x0200
#define DEVICE_CLASS          0      // each configuration has its own class
#define DEVICE_SUB_CLASS      0      // each configuration has its own sub-class
#define DEVICE_PROTOCOL       0      // each configuration has its own protocol
#define EP_CONTROL_LENGTH     64
#define VENDOR_ID             0x1781   // mecanique
#define PRODUCT_ID            0x0BAF
#define RELEASE_NUMBER        0x1000
#define MAN_INDEX             0x01
#define PROD_INDEX            0x02	
#define SN_INDEX              0x03
#define NB_CONFIGURATION      1

               // HID generic CONFIGURATION
#define NB_INTERFACE       1
#define CONF_NB            1
#define CONF_INDEX         0
#define CONF_ATTRIBUTES    USB_CONFIG_BUSPOWERED
#define MAX_POWER          50          // 100 mA

             // USB Interface descriptor
#define INTERFACE_NB        0
#define ALTERNATE           0
#define NB_ENDPOINT         2
#define INTERFACE_CLASS     0xFF    // Vender-specific
#define INTERFACE_SUB_CLASS 0xFF
#define INTERFACE_PROTOCOL  0xFF
#define INTERFACE_INDEX     0

             // USB Endpoint 1 descriptor FS
#define ENDPOINT_BULK_OUT   0x06 // endpoint address (number + OUT)
#define EP_1_ATTRIBUTES     0x02          // BULK = 0x02, INTERUPT = 0x03, see also usb_specific_request.c
#define LARGE_BUFFER
#ifdef LARGE_BUFFER
#define EP_1_MAX_LENGTH     16 // num bytes
#define EP_1_MAX_LENGTH_CODE SIZE_16 // num bytes
#else
#define EP_1_MAX_LENGTH     8 // num bytes
#define EP_1_MAX_LENGTH_CODE SIZE_8 // num bytes
#endif
#define EP_1_INTERVAL       0 // maximum NAKs per (micro)frame (Does this firmware send NAKs?)

             // USB Endpoint 2 descriptor FS
#define ENDPOINT_BULK_IN    0x82 // endpoint address (number + IN)
#define EP_2_ATTRIBUTES     0x02          // BULK = 0x02, INTERUPT = 0x03, see also usb_specific_request.c
#ifdef LARGE_BUFFER
#define EP_2_MAX_LENGTH     16 // num bytes
#define EP_2_MAX_LENGTH_CODE SIZE_16 // num bytes
#else
#define EP_2_MAX_LENGTH     8 // num bytes
#define EP_2_MAX_LENGTH_CODE SIZE_8 // num bytes
#endif
#define EP_2_INTERVAL       0 // maximum NAKs per (micro)frame (Does this firmware send NAKs?)

/*
#define SIZE_OF_REPORT        0x35
#define LENGTH_OF_REPORT_IN      0x08
#define LENGTH_OF_REPORT_OUT     0x08
*/

#define DEVICE_STATUS         0x00 // TBD
#define INTERFACE_STATUS      0x00 // TBD

#define LANG_ID               0x00


/* HID specific */
#define HID                   0x21
#define REPORT                0x22
#define SET_REPORT	      0x02

#define HID_DESCRIPTOR        0x21
#define HID_BCD               0x1001
#define HID_COUNTRY_CODE      0x00 // Not localized
#define HID_CLASS_DESC_NB     0x01
#define HID_DESCRIPTOR_TYPE   0x22


#define USB_MN_LENGTH         8
#define USB_MANUFACTURER_NAME \
{ Usb_unicode('S') \
, Usb_unicode('t') \
, Usb_unicode('r') \
, Usb_unicode('a') \
, Usb_unicode('w') \
, Usb_unicode('m') \
, Usb_unicode('a') \
, Usb_unicode('n') \
}

#define USB_PN_LENGTH         22
#define USB_PRODUCT_NAME \
{ Usb_unicode('F') \
 ,Usb_unicode('l') \
 ,Usb_unicode('y') \
 ,Usb_unicode('d') \
 ,Usb_unicode('r') \
 ,Usb_unicode('a') \
 ,Usb_unicode(' ') \
 ,Usb_unicode('T') \
 ,Usb_unicode('r') \
 ,Usb_unicode('i') \
 ,Usb_unicode('g') \
 ,Usb_unicode('g') \
 ,Usb_unicode('e') \
 ,Usb_unicode('r') \
 ,Usb_unicode(' ') \
 ,Usb_unicode('C') \
 ,Usb_unicode('o') \
 ,Usb_unicode('n') \
 ,Usb_unicode('t') \
 ,Usb_unicode('r') \
 ,Usb_unicode('o') \
 ,Usb_unicode('l') \
}

#define USB_SN_LENGTH         0x05
#define USB_SERIAL_NUMBER \
{ Usb_unicode('1') \
 ,Usb_unicode('.') \
 ,Usb_unicode('0') \
 ,Usb_unicode('.') \
 ,Usb_unicode('0') \
}

// English - United States
#define LANGUAGE_ID           0x0409


                  //! Usb Request
typedef struct
{
   uint8_t      bmRequestType;        //!< Characteristics of the request
   uint8_t      bRequest;             //!< Specific request
   uint16_t     wValue;               //!< field that varies according to request
   uint16_t     wIndex;               //!< field that varies according to request
   uint16_t     wLength;              //!< Number of bytes to transfer if Data
}  S_UsbRequest;

                //! Usb Device Descriptor
typedef struct {
   uint8_t      bLength;              //!< Size of this descriptor in bytes
   uint8_t      bDescriptorType;      //!< DEVICE descriptor type
   uint16_t     bscUSB;               //!< Binay Coded Decimal Spec. release
   uint8_t      bDeviceClass;         //!< Class code assigned by the USB
   uint8_t      bDeviceSubClass;      //!< Sub-class code assigned by the USB
   uint8_t      bDeviceProtocol;      //!< Protocol code assigned by the USB
   uint8_t      bMaxPacketSize0;      //!< Max packet size for EP0
   uint16_t     idVendor;             //!< Vendor ID. ATMEL = 0x03EB
   uint16_t     idProduct;            //!< Product ID assigned by the manufacturer
   uint16_t     bcdDevice;            //!< Device release number
   uint8_t      iManufacturer;        //!< Index of manu. string descriptor
   uint8_t      iProduct;             //!< Index of prod. string descriptor
   uint8_t      iSerialNumber;        //!< Index of S.N.  string descriptor
   uint8_t      bNumConfigurations;   //!< Number of possible configurations
}  S_usb_device_descriptor;


          //! Usb Configuration Descriptor
typedef struct {
   uint8_t      bLength;              //!< size of this descriptor in bytes
   uint8_t      bDescriptorType;      //!< CONFIGURATION descriptor type
   uint16_t     wTotalLength;         //!< total length of data returned
   uint8_t      bNumInterfaces;       //!< number of interfaces for this conf.
   uint8_t      bConfigurationValue;  //!< value for SetConfiguration resquest
   uint8_t      iConfiguration;       //!< index of string descriptor
   uint8_t      bmAttibutes;          //!< Configuration characteristics
   uint8_t      MaxPower;             //!< maximum power consumption
}  S_usb_configuration_descriptor;


              //! Usb Interface Descriptor
typedef struct {
   uint8_t      bLength;               //!< size of this descriptor in bytes
   uint8_t      bDescriptorType;       //!< INTERFACE descriptor type
   uint8_t      bInterfaceNumber;      //!< Number of interface
   uint8_t      bAlternateSetting;     //!< value to select alternate setting
   uint8_t      bNumEndpoints;         //!< Number of EP except EP 0
   uint8_t      bInterfaceClass;       //!< Class code assigned by the USB
   uint8_t      bInterfaceSubClass;    //!< Sub-class code assigned by the USB
   uint8_t      bInterfaceProtocol;    //!< Protocol code assigned by the USB
   uint8_t      iInterface;            //!< Index of string descriptor
}  S_usb_interface_descriptor;


               //! Usb Endpoint Descriptor
typedef struct {
   uint8_t      bLength;               //!< Size of this descriptor in bytes
   uint8_t      bDescriptorType;       //!< ENDPOINT descriptor type
   uint8_t      bEndpointAddress;      //!< Address of the endpoint
   uint8_t      bmAttributes;          //!< Endpoint's attributes
   uint16_t     wMaxPacketSize;        //!< Maximum packet size for this EP
   uint8_t      bInterval;             //!< Interval for polling EP in ms
} S_usb_endpoint_descriptor;


               //! Usb Device Qualifier Descriptor
typedef struct {
   uint8_t      bLength;               //!< Size of this descriptor in bytes
   uint8_t      bDescriptorType;       //!< Device Qualifier descriptor type
   uint16_t     bscUSB;                //!< Binay Coded Decimal Spec. release
   uint8_t      bDeviceClass;          //!< Class code assigned by the USB
   uint8_t      bDeviceSubClass;       //!< Sub-class code assigned by the USB
   uint8_t      bDeviceProtocol;       //!< Protocol code assigned by the USB
   uint8_t      bMaxPacketSize0;       //!< Max packet size for EP0
   uint8_t      bNumConfigurations;    //!< Number of possible configurations
   uint8_t      bReserved;             //!< Reserved for future use, must be zero
}  S_usb_device_qualifier_descriptor;


               //! Usb Language Descriptor
typedef struct {
   uint8_t      bLength;               //!< size of this descriptor in bytes
   uint8_t      bDescriptorType;       //!< STRING descriptor type
   uint16_t     wlangid;               //!< language id
}  S_usb_language_id;


//_____ U S B   M A N U F A C T U R E R   D E S C R I P T O R _______________


//struct usb_st_manufacturer
typedef struct {
   uint8_t  bLength;               // size of this descriptor in bytes
   uint8_t  bDescriptorType;       // STRING descriptor type
   uint16_t wstring[USB_MN_LENGTH];// unicode characters
} S_usb_manufacturer_string_descriptor;


//_____ U S B   P R O D U C T   D E S C R I P T O R _________________________


//struct usb_st_product
typedef struct {
   uint8_t  bLength;               // size of this descriptor in bytes
   uint8_t  bDescriptorType;       // STRING descriptor type
   uint16_t wstring[USB_PN_LENGTH];// unicode characters
} S_usb_product_string_descriptor;


//_____ U S B   S E R I A L   N U M B E R   D E S C R I P T O R _____________


//struct usb_st_serial_number
typedef struct {
   uint8_t  bLength;               // size of this descriptor in bytes
   uint8_t  bDescriptorType;       // STRING descriptor type
   uint16_t wstring[USB_SN_LENGTH];// unicode characters
} S_usb_serial_number;


/*_____ U S B   H I D   D E S C R I P T O R __________________________________*/

typedef struct {
  uint8_t  bLength;               /* Size of this descriptor in bytes */
  uint8_t  bDescriptorType;       /* HID descriptor type */
  uint16_t bscHID;                /* Binay Coded Decimal Spec. release */
  uint8_t  bCountryCode;          /* Hardware target country */
  uint8_t  bNumDescriptors;       /* Number of HID class descriptors to follow */
  uint8_t  bRDescriptorType;      /* Report descriptor type */
  uint16_t wDescriptorLength;     /* Total length of Report descriptor */
} S_usb_hid_descriptor;






typedef struct
{
   S_usb_configuration_descriptor cfg;
   S_usb_interface_descriptor     ifc;
   S_usb_endpoint_descriptor      ep1;
   S_usb_endpoint_descriptor      ep2;
} S_usb_user_configuration_descriptor;




#endif

