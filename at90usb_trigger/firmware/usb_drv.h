//! @file $RCSfile: usb_drv.h,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Use of this program is subject to Atmel's End User License Agreement.
//! Please read file license.txt for copyright notice.
//!
//! @brief This file contains the USB low level driver definition
//!
//! @version $Revision: 1.13 $ $Name: at90usb128-demo-hidgen-last $ $Id: usb_drv.h,v 1.13 2006/01/17 15:26:30 rletendu Exp $
//!
//! @todo
//! @bug


#ifndef _USB_DRV_H_
#define _USB_DRV_H_

//_____ I N C L U D E S ____________________________________________________


typedef enum endpoint_parameter{ep_num, ep_type, ep_direction, ep_size, ep_bank, nyet_status} t_endpoint_parameter;

//! @defgroup USB AT90USBXXX driver Module
//! USB low level drivers Module
//! @{

//_____ M A C R O S ________________________________________________________

#define EP_CONTROL            0
#define EP_1                  1
#define EP_2                  2
#define EP_3                  3
#define EP_4                  4
#define EP_5                  5
#define EP_6                  6
#define EP_7                  7

#define PIPE_CONTROL          0
#define PIPE_0                0
#define PIPE_1                1
#define PIPE_2                2
#define PIPE_3                3
#define PIPE_4                4
#define PIPE_5                5
#define PIPE_6                6
#define PIPE_7                7

//! USB EndPoint
#define MSK_EP_DIR            0x7F
#define MSK_UADD              0x7F
#define MSK_EPTYPE            0xC0
#define MSK_EPSIZE            0x70
#define MSK_EPBK              0x0C
#define MSK_DTSEQ             0x0C
#define MSK_NBUSYBK           0x03
#define MSK_CURRBK            0x03
#define MSK_DAT               0xFF  // UEDATX
#define MSK_BYCTH             0x07  // UEBCHX
#define MSK_BYCTL             0xFF  // UEBCLX
#define MSK_EPINT             0x7F  // UEINT
#define MSK_HADDR             0xFF  // UHADDR

//! USB Pipe
#define MSK_PNUM              0x07  // UPNUM
#define MSK_PRST              0x7F  // UPRST
#define MSK_PTYPE             0xC0  // UPCFG0X
#define MSK_PTOKEN            0x30
#define MSK_PEPNUM            0x0F
#define MSK_PSIZE             0x70  // UPCFG1X
#define MSK_PBK               0x0C

#define MSK_ERROR             0x1F

#define MSK_PTYPE             0xC0  // UPCFG0X
#define MSK_PTOKEN            0x30
#define MSK_TOKEN_SETUP       0x30
#define MSK_TOKEN_IN          0x10
#define MSK_TOKEN_OUT         0x20
#define MSK_PEPNUM            0x0F

#define MSK_PSIZE             0x70  // UPCFG1X
#define MSK_PBK               0x0C


//! Parameters for endpoint configuration
//! These define are the values used to enable and configure an endpoint.
#define TYPE_CONTROL             0
#define TYPE_ISOCHRONOUS         1
#define TYPE_BULK                2
#define TYPE_INTERRUPT           3
 //typedef enum ep_type {TYPE_CONTROL, TYPE_BULK, TYPE_ISOCHRONOUS, TYPE_INTERRUPT} e_ep_type;

#define DIRECTION_OUT            0
#define DIRECTION_IN             1
 //typedef enum ep_dir {DIRECTION_OUT, DIRECTION_IN} e_ep_dir;

#define SIZE_8                   0
#define SIZE_16                  1
#define SIZE_32                  2
#define SIZE_64                  3
#define SIZE_128                 4
#define SIZE_256                 5
#define SIZE_512                 6
#define SIZE_1024                7
 //typedef enum ep_size {SIZE_8,   SIZE_16,  SIZE_32,  SIZE_64,
 //                      SIZE_128, SIZE_256, SIZE_512, SIZE_1024} e_ep_size;

#define ONE_BANK                 0
#define TWO_BANKS                1
 //typedef enum ep_bank {ONE_BANK, TWO_BANKS} e_ep_bank;

#define NYET_ENABLED             0
#define NYET_DISABLED            1
 //typedef enum ep_nyet {NYET_DISABLED, NYET_ENABLED} e_ep_nyet;

#define TOKEN_SETUP              0
#define TOKEN_IN                 1
#define TOKEN_OUT                2




//! @defgroup USB_configuration USB configuration
//! List of the standard macro used to configure pipes and endpoints
//! @{
#define Usb_build_ep_config0(type, dir, nyet)     ((type<<6) | (nyet<<1) | (dir))
#define Usb_build_ep_config1(size, bank     )     ((size<<4) | (bank<<2)        )
#define usb_configure_endpoint(num, type, dir, size, bank, nyet)             \
                                    ( Usb_select_endpoint(num),              \
                                      usb_config_ep(Usb_build_ep_config0(type, dir, nyet),\
                                                    Usb_build_ep_config1(size, bank)    ))

#define Host_build_pipe_config0(type, token, ep_num)     ((type<<6) | (token<<4) | (ep_num))
#define Host_build_pipe_config1(size, bank     )         ((size<<4) | (bank<<2)        )
#define host_configure_pipe(num, type, token,ep_num, size, bank, freq)             \
                                    ( Host_select_pipe(num),              \
                                      Host_set_interrupt_frequency(freq), \
                                      host_config_pipe(Host_build_pipe_config0(type, token, ep_num),\
                                                       Host_build_pipe_config1(size, bank)    ))
//! @}

//! @defgroup USB Pads Regulator Management
//! Turns ON/OFF USB pads regulator
//! @{
#define Usb_enable_regulator()          (UHWCON |= (1<<UVREGE))      //!< Enable internal USB pads regulator
#define Usb_disable_regulator()         (UHWCON &= ~(1<<UVREGE))     //!< Disable internal USB pads regulator
#define Is_usb_regulator_enabled()      ((UHWCON &  (1<<UVREGE))  ? TRUE : FALSE) //!< Check regulator enable bit
//! @}

//! @defgroup General USB management
//! These macros manage the USB controller
//! @{
#define Usb_enable_uid_pin()            (UHWCON |= (1<<UIDE))                         //!< Enable external UID pin
#define Usb_disable_uid_pin()           (UHWCON &= ~(1<<UIDE))                        //!< Disable external UID pin
#define Usb_force_device_mode()         (Usb_disable_uid_pin(), UHWCON |= (1<<UIMOD)) //!< Disable external UID pin and force device mode
#define Usb_force_host_mode()           (Usb_disable_uid_pin(), UHWCON &= ~(1<<UIMOD))//!< Disable external UID pin and force host mode
#define Usb_enable_uvcon_pin()          (UHWCON |= (1<<UVCONE))                       //!< Enable external UVCON pin
#define Usb_full_speed_mode()           (UHWCON |= (1<<UDSS))                         //!< Disable external UVCON pin
#define Usb_low_speed_mode()            (UHWCON &= ~(1<<UDSS))                        //!< For device mode, select USB low speed mode

#define Usb_enable()                  (USBCON |= ((1<<USBE) | (1<<OTGPADE)))          //!< Enable both USB interface and Vbus pad
#define Usb_disable()                 (USBCON &= ~((1<<USBE) | (1<<OTGPADE)))         //!< Disable both USB interface and Vbus pad
#define Is_usb_enabled()              ((USBCON  &   (1<<USBE))   ? TRUE : FALSE)

#define Usb_enable_vbus_pad()         (USBCON |= (1<<OTGPADE))                        //!< Enable VBUS pad
#define Usb_disable_vbus_pad()        (USBCON &= (1<<OTGPADE))                        //!< Disable VBUS pad

#define Usb_select_device()           (USBCON  &= ~(1<<HOST))
#define Usb_select_host()             (USBCON  |=  (1<<HOST))
#define Is_usb_host_enabled()         ((USBCON  &   (1<<HOST))   ? TRUE : FALSE)

#define Usb_freeze_clock()            (USBCON  |=  (1<<FRZCLK))
#define Usb_unfreeze_clock()          (USBCON  &= ~(1<<FRZCLK))
#define Is_usb_clock_freezed()        ((USBCON  &   (1<<FRZCLK)) ? TRUE : FALSE)

#define Usb_enable_id_interrupt()     (USBCON  |=  (1<<IDTE))
#define Usb_disable_id_interrupt()    (USBCON  &= ~(1<<IDTE))
#define Is_usb_id_interrrupt_enabled() ((USBCON &  (1<<IDTE))     ? TRUE : FALSE)
#define Is_usb_id_device()            ((USBSTA &   (1<<ID))      ? TRUE : FALSE)
#define Usb_ack_id_transition()       (USBINT  = ~(1<<IDTI))
#define Is_usb_id_transition()        ((USBINT &   (1<<IDTI))    ? TRUE : FALSE)


#define Usb_enable_vbus_interrupt()   (USBCON  |=  (1<<VBUSTE))
#define Usb_disable_vbus_interrupt()  (USBCON  &= ~(1<<VBUSTE))
#define Is_usb_vbus_interrrupt_enabled() ((USBCON &  (1<<VBUSTE))     ? TRUE : FALSE)
#define Is_usb_vbus_high()            ((USBSTA &   (1<<VBUS))    ? TRUE : FALSE)
#define Usb_ack_vbus_transition()     (USBINT  = ~(1<<VBUSTI))
#define Is_usb_vbus_transition()      ((USBINT &   (1<<VBUSTI))  ? TRUE : FALSE)



#define Usb_get_general_interrupt()      (USBINT & (USBCON & MSK_IDTE_VBUSTE))   //!< returns the USB general interrupts (interrupt enabled)
#define Usb_ack_all_general_interrupt()  (USBINT = ~(USBCON & MSK_IDTE_VBUSTE))  //!< acks the general interrupts (interrupt enabled)
#define Usb_ack_cache_id_transition(x)   ((x)  &= ~(1<<IDTI))
#define Usb_ack_cache_vbus_transition(x) ((x)  &= ~(1<<VBUSTI))
#define Is_usb_cache_id_transition(x)    (((x) &   (1<<IDTI))  )
#define Is_usb_cache_vbus_transition(x)  (((x) &   (1<<VBUSTI)))

#define Usb_get_otg_interrupt()            (OTGINT & OTGIEN)                     //!< returns the USB Pad interrupts (interrupt enabled)
#define Usb_ack_all_otg_interrupt()        (OTGINT = ~OTGIEN)                    //!< acks the USB Pad interrupts (interrupt enabled)
#define Is_otg_cache_bconnection_error(x)  (((x) &   MSK_BCERRI))
#define Usb_ack_cache_bconnection_error(x) ((x)  &= ~MSK_BCERRI)

#define Usb_enter_dpram_mode()        (UDPADDH =  (1<<DPACC))
#define Usb_exit_dpram_mode()         (UDPADDH =  (uint8_t)~(1<<DPACC))
#define Usb_set_dpram_address(addr)   (UDPADDH =  (1<<DPACC) + ((Uint16)addr >> 8), UDPADDL = (Uchar)addr)
#define Usb_write_dpram_byte(val)     (UEDATX=val)
#define Usb_read_dpram_byte()			  (UEDATX)

#define Usb_enable_vbus()             (OTGCON  |=  (1<<VBUSREQ))                             //!< requests for VBus activation
#define Usb_disable_vbus()            (OTGCON  &= ~(1<<VBUSREQ))                             //!< requests for VBus desactivation
#define Usb_device_initiate_hnp()     (OTGCON  |=  (1<<HNPREQ))                              //!< initiates a Host Negociation Protocol
#define Usb_host_accept_hnp()         (OTGCON  |=  (1<<HNPREQ))                              //!< accepts a Host Negociation Protocol
#define Usb_host_reject_hnp()         (OTGCON  &= ~(1<<HNPREQ))                              //!< rejects a Host Negociation Protocol
#define Usb_device_initiate_srp()     (OTGCON  |=  (1<<SRPREQ))                              //!< initiates a Session Request Protocol
#define Usb_select_vbus_srp_method()  (OTGCON  |=  (1<<SRPSEL))                              //!< selects VBus as SRP method
#define Usb_select_data_srp_method()  (OTGCON  &= ~(1<<SRPSEL))                              //!< selects data line as SRP method
#define Usb_enable_vbus_hw_control()  (OTGCON  &= ~(1<<VBUSHWC))                             //!< enables hardware control on VBus
#define Usb_disable_vbus_hw_control() (OTGCON  |=  (1<<VBUSHWC))                             //!< disables hardware control on VBus
#define Is_usb_vbus_enabled()         ((OTGCON &   (1<<VBUSREQ)) ? TRUE : FALSE)             //!< tests if VBus has been requested
#define Is_usb_hnp()                  ((OTGCON &   (1<<HNPREQ))  ? TRUE : FALSE)             //!< tests if a HNP occurs
#define Is_usb_device_srp()           ((OTGCON &   (1<<SRPREQ))  ? TRUE : FALSE)             //!< tests if a SRP from device occurs

#define Usb_enable_suspend_time_out_interrupt()   (OTGIEN  |=  (1<<STOE))                    //!< enables suspend time out interrupt
#define Usb_disable_suspend_time_out_interrupt()  (OTGIEN  &= ~(1<<STOE))                    //!< disables suspend time out interrupt
#define Is_suspend_time_out_interrupt_enabled()   ((OTGIEN &  (1<<STOE))   ? TRUE : FALSE)
#define Usb_ack_suspend_time_out_interrupt()      (OTGINT  &= ~(1<<STOI))                    //!< acks suspend time out interrupt
#define Is_usb_suspend_time_out_interrupt()       ((OTGINT &   (1<<STOI))    ? TRUE : FALSE) //!< tests if a suspend time out occurs

#define Usb_enable_hnp_error_interrupt()          (OTGIEN  |=  (1<<HNPERRE))                 //!< enables HNP error interrupt
#define Usb_disable_hnp_error_interrupt()         (OTGIEN  &= ~(1<<HNPERRE))                 //!< disables HNP error interrupt
#define Is_hnp_error_interrupt_enabled()          ((OTGIEN &  (1<<HNPERRE))   ? TRUE : FALSE)
#define Usb_ack_hnp_error_interrupt()             (OTGINT  &= ~(1<<HNPERRI))                 //!< acks HNP error interrupt
#define Is_usb_hnp_error_interrupt()              ((OTGINT &   (1<<HNPERRI)) ? TRUE : FALSE) //!< tests if a HNP error occurs

#define Usb_enable_role_exchange_interrupt()      (OTGIEN  |=  (1<<ROLEEXE))                 //!< enables role exchange interrupt
#define Usb_disable_role_exchange_interrupt()     (OTGIEN  &= ~(1<<ROLEEXE))                 //!< disables role exchange interrupt
#define Is_role_exchange_interrupt_enabled()      ((OTGIEN &  (1<<ROLEEXE))   ? TRUE : FALSE)
#define Usb_ack_role_exchange_interrupt()         (OTGINT  &= ~(1<<ROLEEXI))                 //!< acks role exchange interrupt
#define Is_usb_role_exchange_interrupt()          ((OTGINT &   (1<<ROLEEXI)) ? TRUE : FALSE) //!< tests if a role exchange occurs

#define Usb_enable_bconnection_error_interrupt()  (OTGIEN  |=  (1<<BCERRE))                  //!< enables B device connection error interrupt
#define Usb_disable_bconnection_error_interrupt() (OTGIEN  &= ~(1<<BCERRE))                  //!< disables B device connection error interrupt
#define Is_bconnection_error_interrupt_enabled()  ((OTGIEN &  (1<<BCERRE))   ? TRUE : FALSE)
#define Usb_ack_bconnection_error_interrupt()     (OTGINT  &= ~(1<<BCERRI))                  //!< acks B device connection error interrupt
#define Is_usb_bconnection_error_interrupt()      ((OTGINT &   (1<<BCERRI))  ? TRUE : FALSE) //!< tests if a B device connection error occurs

#define Usb_enable_vbus_error_interrupt()         (OTGIEN  |=  (1<<VBERRE))                  //!< enables VBus error interrupt
#define Usb_disable_vbus_error_interrupt()        (OTGIEN  &= ~(1<<VBERRE))                  //!< disables VBus error interrupt
#define Is_vbus_error_interrupt_enabled()         ((OTGIEN &  (1<<VBERRE))   ? TRUE : FALSE)
#define Usb_ack_vbus_error_interrupt()            (OTGINT  &= ~(1<<VBERRI))                  //!< acks VBus error interrupt
#define Is_usb_vbus_error_interrupt()             ((OTGINT &   (1<<VBERRI))  ? TRUE : FALSE) //!< tests if a VBus error occurs

#define Usb_enable_srp_interrupt()                (OTGIEN  |=  (1<<SRPE))                    //!< enables SRP interrupt
#define Usb_disable_srp_interrupt()               (OTGIEN  &= ~(1<<SRPE))                    //!< disables SRP interrupt
#define Is_srp_interrupt_enabled()                ((OTGIEN &  (1<<SRPE))   ? TRUE : FALSE)
#define Usb_ack_srp_interrupt()                   (OTGINT  &= ~(1<<SRPI))                    //!< acks SRP interrupt
#define Is_usb_srp_interrupt()                    ((OTGINT &   (1<<SRPI))    ? TRUE : FALSE) //!< tests if a SRP occurs
//! @}


//! @defgroup USB Device management
//! These macros manage the USB Device controller.
//! @{
#define Usb_initiate_remote_wake_up()             (UDCON   |=  (1<<RMWKUP))                  //!< initiates a remote wake-up
#define Usb_detach()                              (UDCON   |=  (1<<DETACH))                  //!< detaches from USB bus
#define Usb_attach()                              (UDCON   &= ~(1<<DETACH))                  //!< attaches to USB bus
#define Is_usb_pending_remote_wake_up()           ((UDCON & (1<<RMWKUP)) ? TRUE : FALSE)     //!< test if remote wake-up still running
#define Is_usb_detached()                         ((UDCON & (1<<DETACH)) ? TRUE : FALSE)     //!< test if the device is detached

#define Usb_get_device_interrupt()                (UDINT   &   (1<<UDIEN))                   //!< returns the USB device interrupts (interrupt enabled)
#define Usb_ack_all_device_interrupt()            (UDINT   =  ~(1<<UDIEN))                   //!< acks the USB device interrupts (interrupt enabled)

#define Usb_ack_cache_remote_wake_up_start(x)     ((x) &= ~(1<<UPRSMI))
#define Usb_ack_cache_resume(x)                   ((x) &= ~(1<<EORSMI) )
#define Usb_ack_cache_wake_up(x)                  ((x) &= ~(1<<WAKEUPI))
#define Usb_ack_cache_reset(x)                    ((x) &= ~(1<<EORSTI) )
#define Usb_ack_cache_sof(x)                      ((x) &= ~(1<<SOFI)   )
#define Usb_ack_cache_micro_sof(x)                ((x) &= ~(1<<MSOFI)  )
#define Usb_ack_cache_suspend(x)                  ((x) &= ~(1<<SUSPI)  )
#define Is_usb_cache_remote_wake_up_start(x)      ((x) & (1<<UPRSMI)   )
#define Is_usb_cache_resume(x)                    ((x) & (1<<EORSMI)   )
#define Is_usb_cache_wake_up(x)                   ((x) & (1<<WAKEUPI)  )
#define Is_usb_cache_reset(x)                     ((x) & (1<<EORSTI)   )
#define Is_usb_cache_sof(x)                       ((x) & (1<<SOFI)     )
#define Is_usb_cache_suspend(x)                   ((x) & (1<<SUSPI)    )


#define Usb_enable_remote_wake_up_interrupt()     (UDIEN   |=  (1<<UPRSME))                  //!< enables remote wake-up interrupt
#define Usb_disable_remote_wake_up_interrupt()    (UDIEN   &= ~(1<<UPRSME))                  //!< disables remote wake-up interrupt
#define Is_remote_wake_up_interrupt_enabled()     ((UDIEN &  (1<<UPRSME))   ? TRUE : FALSE)
#define Usb_ack_remote_wake_up_start()            (UDINT   = ~(1<<UPRSMI))                   //!< acks remote wake-up
#define Is_usb_remote_wake_up_start()             ((UDINT &   (1<<UPRSMI))  ? TRUE : FALSE)  //!< tests if remote wake-up still running

#define Usb_enable_resume_interrupt()             (UDIEN   |=  (1<<EORSME))                  //!< enables resume interrupt
#define Usb_disable_resume_interrupt()            (UDIEN   &= ~(1<<EORSME))                  //!< disables resume interrupt
#define Is_resume_interrupt_enabled()             ((UDIEN &  (1<<EORSME))   ? TRUE : FALSE)
#define Usb_ack_resume()                          (UDINT   = ~(1<<EORSMI))                   //!< acks resume
#define Is_usb_resume()                           ((UDINT &   (1<<EORSMI))  ? TRUE : FALSE)  //!< tests if resume occurs

#define Usb_enable_wake_up_interrupt()            (UDIEN   |=  (1<<WAKEUPE))                 //!< enables wake-up interrupt
#define Usb_disable_wake_up_interrupt()           (UDIEN   &= ~(1<<WAKEUPE))                 //!< disables wake-up interrupt
#define Is_swake_up_interrupt_enabled()           ((UDIEN &  (1<<WAKEUPE))   ? TRUE : FALSE)
#define Usb_ack_wake_up()                         (UDINT   = ~(1<<WAKEUPI))                  //!< acks wake-up
#define Is_usb_wake_up()                          ((UDINT &   (1<<WAKEUPI)) ? TRUE : FALSE)  //!< tests if wake-up occurs

#define Usb_enable_reset_interrupt()              (UDIEN   |=  (1<<EORSTE))                  //!< enables USB reset interrupt
#define Usb_disable_reset_interrupt()             (UDIEN   &= ~(1<<EORSTE))                  //!< disables USB reset interrupt
#define Is_reset_interrupt_enabled()              ((UDIEN &  (1<<EORSTE))   ? TRUE : FALSE)
#define Usb_ack_reset()                           (UDINT   = ~(1<<EORSTI))                   //!< acks USB reset
#define Is_usb_reset()                            ((UDINT &   (1<<EORSTI))  ? TRUE : FALSE)  //!< tests if USB reset occurs

#define Usb_enable_sof_interrupt()                (UDIEN   |=  (1<<SOFE))                    //!< enables Start Of Frame Interrupt
#define Usb_disable_sof_interrupt()               (UDIEN   &= ~(1<<SOFE))                    //!< disables Start Of Frame Interrupt
#define Is_sof_interrupt_enabled()                ((UDIEN &  (1<<SOFE))   ? TRUE : FALSE)
#define Usb_ack_sof()                             (UDINT   = ~(1<<SOFI))                     //!< acks Start Of Frame
#define Is_usb_sof()                              ((UDINT &   (1<<SOFI))    ? TRUE : FALSE)  //!< tests if Start Of Frame occurs

#define Usb_enable_suspend_interrupt()            (UDIEN   |=  (1<<SUSPE))                   //!< enables suspend state interrupt
#define Usb_disable_suspend_interrupt()           (UDIEN   &= ~(1<<SUSPE))                   //!< disables suspend state interrupt
#define Is_suspend_interrupt_enabled()            ((UDIEN &  (1<<SUSPE))   ? TRUE : FALSE)
#define Usb_ack_suspend()                         (UDINT   = ~(1<<SUSPI))                    //!< acks Suspend
#define Is_usb_suspend()                          ((UDINT &   (1<<SUSPI))   ? TRUE : FALSE)  //!< tests if Suspend state detected

#define Usb_enable_address()                      (UDADDR  |=  (1<<ADDEN))                                     //!< enables USB device address
#define Usb_disable_address()                     (UDADDR  &= ~(1<<ADDEN))                                     //!< disables USB device address
#define Usb_configure_address(addr)               (UDADDR  =   (UDADDR & (1<<ADDEN)) | ((uint8_t)addr & MSK_UADD))  //!< sets the USB device address

#define Usb_frame_number()                        ((uint16_t)((((uint16_t)UDFNUMH) << 8) | ((uint16_t)UDFNUML)))              //!< returns the last frame number
#define Is_usb_frame_number_crc_error()           ((UDMFN & (1<<FNCERR)) ? TRUE : FALSE)                       //!< tests if a crc error occurs in frame number
//! @}




//! @defgroup General endpoint management
//! These macros manage the common features of the endpoints.
//! @{
#define Usb_select_endpoint(ep)                   (UENUM = (uint8_t)ep )                                            //!< selects the endpoint number to interface with the CPU

#define Usb_reset_endpoint(ep)                    (UERST   =   1 << (uint8_t)ep, UERST  =  0)                       //!< resets the selected endpoint

#define Usb_enable_endpoint()                     (UECONX  |=  (1<<EPEN))                                      //!< enables the current endpoint
#define Usb_enable_stall_handshake()              (UECONX  |=  (1<<STALLRQ))                                   //!< enables the STALL handshake for the next transaction
#define Usb_reset_data_toggle()                   (UECONX  |=  (1<<RSTDT))                                     //!< resets the data toggle sequence
#define Usb_disable_endpoint()                    (UECONX  &= ~(1<<EPEN))                                      //!< disables the current endpoint
#define Usb_disable_stall_handshake()             (UECONX  |=  (1<<STALLRQC))                                  //!< desables the STALL handshake
#define Usb_select_epnum_for_cpu()                (UECONX  &= ~(1<<EPNUMS))                                    //!< selects endpoint interface on CPU
#define Is_usb_endpoint_enabled()                 ((UECONX & (1<<EPEN))    ? TRUE : FALSE)                     //!< tests if the current endpoint is enabled
#define Is_usb_endpoint_stall_requested()         ((UECONX & (1<<STALLRQ)) ? TRUE : FALSE)                     //!< tetst if STALL handshake request is running

#define Usb_configure_endpoint_type(type)         (UECFG0X =   (UECFG0X & ~(MSK_EPTYPE)) | ((uint8_t)type << 6))     //!< configures the current endpoint
#define Usb_configure_endpoint_direction(dir)     (UECFG0X =   (UECFG0X & ~(1<<EPDIR))  | ((uint8_t)dir))            //!< configures the current endpoint direction

#define Usb_configure_endpoint_size(size)         (UECFG1X =   (UECFG1X & ~MSK_EPSIZE) | ((uint8_t)size << 4))       //!< configures the current endpoint size
#define Usb_configure_endpoint_bank(bank)         (UECFG1X =   (UECFG1X & ~MSK_EPBK)   | ((uint8_t)bank << 2))       //!< configures the current endpoint number of banks
#define Usb_allocate_memory()                     (UECFG1X |=  (1<<ALLOC))                                      //!< allocates the current configuration in DPRAM memory
#define Usb_unallocate_memory()                   (UECFG1X &= ~(1<<ALLOC))                                      //!< un-allocates the current configuration in DPRAM memory

#define Usb_ack_overflow_interrupt()              (UESTA0X &= ~(1<<OVERFI))                                     //!< acks endpoint overflow interrupt
#define Usb_ack_underflow_interrupt()             (UESTA0X &= ~(1<<UNDERFI))                                    //!< acks endpoint underflow memory
#define Usb_ack_zlp()                             (UESTA0X &= ~(1<<ZLPSEEN))                                    //!< acks Zero Length Packet received
#define Usb_data_toggle()                         ((UESTA0X&MSK_DTSEQ) >> 2)                                    //!< returns data toggle
#define Usb_nb_busy_bank()                        (UESTA0X &   MSK_NBUSYBK)                                     //!< returns the number of busy banks
#define Is_usb_one_bank_busy()                    ((UESTA0X &  MSK_NBUSYBK) == 0 ? FALSE : TRUE)                //!< tests if at least one bank is busy
#define Is_endpoint_configured()                  ((UESTA0X &  (1<<CFGOK))   ? TRUE : FALSE)                    //!< tests if current endpoint is configured
#define Is_usb_overflow()                         ((UESTA0X &  (1<<OVERFI))  ? TRUE : FALSE)                    //!< tests if an overflows occurs
#define Is_usb_underflow()                        ((UESTA0X &  (1<<UNDERFI)) ? TRUE : FALSE)                    //!< tests if an underflow occurs
#define Is_usb_zlp()                              ((UESTA0X &  (1<<ZLPSEEN)) ? TRUE : FALSE)                    //!< tests if a ZLP has been detected

#define Usb_control_direction()                   ((UESTA1X &  (1<<CTRLDIR)) >> 2)                              //!< returns the control directino
#define Usb_current_bank()                        ( UESTA1X & MSK_CURRBK)                                       //!< returns the number of the current bank

#define Usb_ack_fifocon()                         (UEINTX &= ~(1<<FIFOCON))                                     //!< clears FIFOCON bit
#define Usb_ack_nak_in()                          (UEINTX &= ~(1<<NAKINI))                                      //!< acks NAK IN received
#define Usb_ack_nak_out()                         (UEINTX &= ~(1<<NAKOUTI))                                     //!< acks NAK OUT received
#define Usb_ack_receive_setup()                   (UEINTX &= ~(1<<RXSTPI))                                      //!< acks receive SETUP
#define Usb_ack_receive_out()                     (UEINTX &= ~(1<<RXOUTI), Usb_ack_fifocon())                   //!< acks reveive OUT
#define Usb_ack_stalled()                         (MSK_STALLEDI=   0)                                           //!< acks STALL sent
#define Usb_ack_in_ready()                        (UEINTX &= ~(1<<TXINI), Usb_ack_fifocon())                    //!< acks IN ready
#define Usb_kill_last_in_bank()                   (UENTTX |= (1<<RXOUTI))                                       //!< Kills last bank
#define Is_usb_read_enabled()                     (UEINTX&(1<<RWAL))                                            //!< tests if endpoint read allowed
#define Is_usb_write_enabled()                    (UEINTX&(1<<RWAL))                                            //!< tests if endpoint write allowed
#define Is_usb_read_control_enabled()             (UEINTX&(1<<TXINI))                                           //!< tests if read allowed on control endpoint
#define Is_usb_receive_setup()                    (UEINTX&(1<<RXSTPI))                                          //!< tests if SETUP received
#define Is_usb_receive_out()                      (UEINTX&(1<<RXOUTI))                                          //!< tests if OUT received
#define Is_usb_in_ready()                         (UEINTX&(1<<TXINI))                                           //!< tests if IN ready
#define Usb_send_in()                             (UEINTX &= ~(1<<FIFOCON))                                     //!< sends IN
#define Usb_send_control_in()                     (UEINTX &= ~(1<<TXINI))                                       //!< sends IN on control endpoint
#define Usb_free_out_bank()                       (UEINTX &= ~(1<<FIFOCON))                                     //!< frees OUT bank
#define Usb_ack_control_out()                     (UEINTX &= ~(1<<RXOUTI))                                      //!< acks OUT on control endpoint

#define Usb_enable_flow_error_interrupt()         (UEIENX  |=  (1<<FLERRE))                                     //!< enables flow error interrupt
#define Usb_enable_nak_in_interrupt()             (UEIENX  |=  (1<<NAKINE))                                     //!< enables NAK IN interrupt
#define Usb_enable_nak_out_interrupt()            (UEIENX  |=  (1<<NAKOUTE))                                    //!< enables NAK OUT interrupt
#define Usb_enable_receive_setup_interrupt()      (UEIENX  |=  (1<<RXSTPE))                                     //!< enables receive SETUP interrupt
#define Usb_enable_receive_out_interrupt()        (UEIENX  |=  (1<<RXOUTE))                                     //!< enables receive OUT interrupt
#define Usb_enable_stalled_interrupt()            (UEIENX  |=  (1<<STALLEDE))                                   //!< enables STALL sent interrupt
#define Usb_enable_in_ready_interrupt()           (UEIENX  |=  (1<<TXIN))                                       //!< enables IN ready interrupt
#define Usb_disable_flow_error_interrupt()        (UEIENX  &= ~(1<<FLERRE))                                     //!< disables flow error interrupt
#define Usb_disable_nak_in_interrupt()            (UEIENX  &= ~(1<<NAKINE))                                     //!< disables NAK IN interrupt
#define Usb_disable_nak_out_interrupt()           (UEIENX  &= ~(1<<NAKOUTE))                                    //!< disables NAK OUT interrupt
#define Usb_disable_receive_setup_interrupt()     (UEIENX  &= ~(1<<RXSTPE))                                     //!< disables receive SETUP interrupt
#define Usb_disable_receive_out_interrupt()       (UEIENX  &= ~(1<<RXOUTE))                                     //!< disables receive OUT interrupt
#define Usb_disable_stalled_interrupt()           (UEIENX  &= ~(1<<STALLEDE))                                   //!< disables STALL sent interrupt
#define Usb_disable_in_ready_interrupt()          (UEIENX  &= ~(1<<TXIN))                                       //!< disables IN ready interrupt

#define Usb_read_byte()                           (UEDATX)                                                      //!< returns FIFO byte for current endpoint
#define Usb_write_byte(byte)                      (UEDATX  =   (uint8_t)byte)                                        //!< writes byte in FIFO for current endpoint

#define Usb_byte_counter()                        ((((uint16_t)UEBCHX) << 8) | (UEBCLX))                             //!< returns number of bytes in FIFO current endpoint (16 bits)
#define Usb_byte_counter_8()                      ((uint8_t)UEBCLX)                                                  //!< returns number of bytes in FIFO current endpoint (8 bits)

#define Usb_interrupt_flags()                     (UEINT != 0x00)                                               //!< tests the general endpoint interrupt flags
#define Is_usb_endpoint_event()                   (Usb_interrupt_flags())                                       //!< tests the general endpoint interrupt flags

// ADVANCED MACROS
#define Usb_select_ep_for_cpu(ep)                 (Usb_select_epnum_for_cpu(), Usb_select_endpoint(ep))

//! @}



//! @defgroup USB Host management
//! These macros manage the USB Host controller.
//! @{
#define Host_allocate_memory()                 (UPCFG1X |=  (1<<ALLOC))                    //!< allocates the current configuration in DPRAM memory
#define Host_unallocate_memory()               (UPCFG1X &= ~(1<<ALLOC))                    //!< un-allocates the current configuration in DPRAM memory

#define Host_enable()                          (USBCON |= (1<<HOST))                       //!< enables USB Host function
#define Host_enable_sof()                      (UHCON |= (1<<SOFEN))                       //!< enables SOF generation
#define Host_disable_sof()                     (UHCON &= ~(1<<SOFEN))                      //!< disables SOF generation
#define Host_send_reset()                      (UHCON |= (1<<RESET))                       //!< sends a USB Reset to the device
#define Host_is_reset()                        ((UHCON & (1<<RESET)) ? TRUE : FALSE)       //!< tests if USB Reset running
#define Host_send_resume()                     (UHCON |= (1<<RESUME))                      //!< sends a USB Resume to the device
#define Host_is_resume()                       ((UHCON & (1<<RESUME)) ? TRUE : FALSE)      //!< tests if USB Resume running

#define Host_enable_sof_interrupt()            (UHIEN |= (1<<HSOFE))                       //!< enables host start of frame interrupt
#define Host_disable_sof_interrupt()           (UHIEN &= ~(1<<HSOFE))                       //!< enables host start of frame interrupt
#define Is_host_sof_interrupt_enabled()        ((UHIEN &  (1<<HSOFE))   ? TRUE : FALSE)
#define Host_is_sof()                          ((UHINT & (1<<HSOFI)) ? TRUE : FALSE)       //!< tests if SOF detected
#define Is_host_sof()                          ((UHINT & (1<<HSOFI)) ? TRUE : FALSE)
#define Host_ack_sof()                         (UHINT &= ~(1<<HSOFI))

#define Host_enable_device_connection_interrupt()        (UHIEN |= (1<<DCONNE))               //!< enables host device connection interrupt
#define Host_disable_device_connection_interrupt()    (UHIEN &= ~(1<<DCONNE))         //!< disables USB device connection interrupt
#define Is_host_device_connection_interrupt_enabled()    ((UHIEN &  (1<<DCONNE))   ? TRUE : FALSE)
#define Is_device_connection()                 (UHINT & (1<<DCONNI))                       //!< tests if a USB device has been detected
#define Host_ack_device_connection()           (UHINT = ~(1<<DCONNI))                      //!< acks device connection

#define Host_enable_device_disconnection_interrupt()     (UHIEN |= (1<<DDISCE))               //!< enables host device disconnection interrupt
#define Host_disable_device_disconnection_interrupt()    (UHIEN &= ~(1<<DDISCE))         //!< disables USB device connection interrupt
#define Is_host_device_disconnection_interrupt_enabled() ((UHIEN &  (1<<DDISCE))   ? TRUE : FALSE)
#define Is_device_disconnection()              (UHINT & (1<<DDISCI))                       //!< tests if a USB device has been removed
#define Host_ack_device_disconnection()        (UHINT = ~(1<<DDISCI))                      //!< acks device disconnection

#define Host_enable_reset_interrupt()          (UHIEN   |=  (1<<RSTE))                  //!< enables host USB reset interrupt
#define Host_disable_reset_interrupt()         (UHIEN   &= ~(1<<RSTE))                  //!< disables host USB reset interrupt
#define Is_host_reset_interrupt_enabled()      ((UHIEN &  (1<<RSTE))   ? TRUE : FALSE)
#define Host_ack_reset()                       (UHINT   = ~(1<<RSTI))                   //!< acks host USB reset sent
#define Is_host_reset()                        ((UHINT &   (1<<RSTI))  ? TRUE : FALSE)  //!< tests if USB reset has been sent


#define Host_get_host_interrupt()              (UHINT &  UHIEN)                            //!< returns the USB Host interrupts (interrupt enabled)
#define Host_ack_all_host_interrupt()          (UHINT = ~UHIEN)                            //!< acks USB Host interrupts (interrupt enabled)
#define Host_ack_cache_wake_up(x)              ((x) &= ~(1<<HWUPI)  )
#define Host_ack_cache_sof(x)                  ((x) &= ~(1<<HSOFI)  )
#define Host_ack_cache_upstream_resume(x)      ((x) &= ~(1<<RXRSMI) )
#define Host_ack_cache_resume_sent(x)          ((x) &= ~(1<<RSMEDI) )
#define Host_ack_cache_reset_sent(x)           ((x) &= ~(1<<RSTI)   )
#define Host_ack_cache_device_disconnection(x) ((x) &= ~(1<<DDISCI) )
#define Host_ack_cache_device_connection(x)    ((x) &= ~(1<<DCONNI) )
#define Is_host_cache_wake_up(x)               ((x) &   (1<<HWUPI)  )
#define Is_host_cache_sof(x)                   ((x) &   (1<<HSOFI)  )
#define Is_host_cache_upstream_resume(x)       ((x) &   (1<<RXRSMI) )
#define Is_host_cache_resume_sent(x)           ((x) &   (1<<RSMEDI) )
#define Is_host_cache_rreset_sent(x)           ((x) &   (1<<RSTI)   )
#define Is_host_cache_device_disconnection(x)  ((x) &   (1<<DDISCI) )
#define Is_host_cache_device_connection(x)     ((x) &   (1<<DCONNI) )


#define Host_vbus_request()                    (OTGCON |= (1<<VBUSREQ))           //!< switches on VBus
#define Host_clear_vbus_request()              (OTGCON |= (1<<VBUSRQC))           //!< switches off VBus

#define Host_configure_address(addr)           (UHADDR = addr & MSK_HADDR)        //!< configures the address to use for the device
//! @}



//! @defgroup General pipe management
//! These macros manage the common features of the pipes.
//! @{
#define Host_select_pipe(p)                    (UPNUM = (uint8_t)p)                    //!< selects pipe for CPU interface

#define Host_enable_pipe()                     (UPCONX |= (1<<PEN))               //!< enables pipe
#define Host_disable_pipe()                    (UPCONX &= ~(1<<PEN))              //!< disables pipe

#define Host_set_token_setup()                 (UPCFG0X =  UPCFG0X & ~MSK_TOKEN_SETUP)                   //!< sets SETUP token
#define Host_set_token_in()                    (UPCFG0X = (UPCFG0X & ~MSK_TOKEN_SETUP) | MSK_TOKEN_IN)   //!< sets IN token
#define Host_set_token_out()                   (UPCFG0X = (UPCFG0X & ~MSK_TOKEN_SETUP) | MSK_TOKEN_OUT)  //!< sets OUT token

#define Host_get_endpoint_number()             (UPCFG0X & (1<<PEPNUM))            //!< returns the number of the endpoint associated to the current pipe

#define Host_set_interrupt_frequency(frq)      (UPCFG2X = (uint8_t)frq)                //!< sets the interrupt frequency

#define Is_pipe_configured()                   (UPSTAX  &  (1<<CFGOK))            //!< tests if current pipe is configured
#define Is_host_one_bank_busy()                ((UPSTAX &  (1<<NBUSYBK)) != 0)    //!< tests if at least one bank is busy
#define Host_number_of_busy_bank()             (UPSTAX &  (1<<NBUSYBK))           //!< returns the number of busy banks

#define Host_reset_pipe(p)                     (UPRST = 1<<p , UPRST = 0)         //!< resets the pipe

#define Host_write_byte(dat)                   (UPDATX = dat)                     //!< writes a byte into the pipe FIFO
#define Host_read_byte()                       (UPDATX)                           //!< reads a byte from the pipe FIFO

#define Host_freeze_pipe()                     (UPCONX |=  (1<<PFREEZE))          //!< freezes the pipe
#define Host_unfreeze_pipe()                   (UPCONX &= ~(1<<PFREEZE))          //!< un-freezees the pipe
#define Is_host_pipe_freeze()                  (UPCONX &   (1<<PFREEZE))          //!< tests if the current pipe is frozen

#define Host_reset_pipe_data_toggle()          (UPCONX |=  (1<<RSTDT)  )          //!< resets data toggle

#define Is_host_setup_sent()                   ((UPINTX & (1<<TXSTPI))    ? TRUE : FALSE)          //!< tests if SETUP has been sent
#define Is_host_control_in_received()          ((UPINTX & (1<<RXINI))    ? TRUE : FALSE)           //!< tests if control IN has been received
#define Is_host_control_out_sent()             ((UPINTX & (1<<TXOUTI))    ? TRUE : FALSE)          //!< tests if control OUT has been sent
#define Is_host_stall()                        ((UPINTX & (1<<RXSTALLI))    ? TRUE : FALSE)        //!< tests if a STALL has been received
#define Is_host_pipe_error()                   ((UPINTX & (1<<PERRI))    ? TRUE : FALSE)           //!< tests if an error occurs on current pipe
#define Host_send_setup()                      (UPINTX  &= ~(1<<FIFOCON))        //!< sends a setup
#define Host_send_control_in()                 (UPINTX  &= ~(1<<FIFOCON))        //!< sends a control IN
#define Host_send_control_out()                (UPINTX  &= ~(1<<FIFOCON))        //!< sends a control OUT
#define Host_ack_control_out()                 (UPINTX  &= ~(1<<TXOUTI))         //!< acks control OUT
#define Host_ack_control_in()                  (UPINTX  &= ~(1<<RXINI))          //!< acks control IN
#define Host_ack_setup()                       (UPINTX  &= ~(1<<TXOUTI))         //!< acks setup
#define Host_ack_stall()                       (UPINTX  &= ~(1<<RXSTALLI))       //!< acks STALL reception

#define Host_send_out()                        (UPINTX = 0x7B)                   //!< sends a OUT
#define Is_host_out_sent()                     ((UPINTX & (1<<TXOUTI))    ? TRUE : FALSE)   //!< tests if OUT has been sent
#define Host_ack_out_sent()                    (UPINTX = 0xFB)                   //!< acks OUT sent
#define Is_host_in_received()                  ((UPINTX & (1<<FIFOCON))    ? TRUE : FALSE) //!< tests if IN received
#define Host_ack_in_received()                 (UPINTX = 0x7E)                   //!< acks IN reception

#define Host_standard_in_mode()                (UPCONX &= ~(1<<INMODE))          //!< sets IN in standard mode
#define Host_continuous_in_mode()              (UPCONX |=  (1<<INMODE))          //!< sets IN in continuous mode

#define Host_in_request_number(in_num)         (UPINRQX = (uint8_t)in_num)            //!< sets number of IN requests to perform before freeze
#define Host_get_in_request_number()           (UPINRQX)                         //!< returns number of remaining IN requests

#define Host_data_length_uint8_t()                  (UPBCLX)                          //!< returns number of bytes (8 bits)
#define Host_data_length_uint16_t()                 ((((uint16_t)UPBCHX)<<8) | UPBCLX)     //!< returns number of bytes (16 bits)

#define Host_get_pipe_length()                 ((uint16_t)0x08 << ((UPCFG1X & MSK_PSIZE)>>4))  //!< returns the size of the current pipe

#define Host_error_status()                    (UPERRX & MSK_ERROR)              //!< tests if error occurs on pipe
#define Host_ack_all_errors()                  (UPERRX = 0x00)                   //!< acks all pipe error


#define Host_set_device_supported()   (device_status |=  0x01)
#define Host_clear_device_supported() (device_status &= ~0x01)
#define Is_host_device_supported()    (device_status &   0x01)

#define Host_set_device_ready()       (device_status |=  0x02)
#define Host_clear_device_ready()     (device_status &= ~0x02)
#define Is_host_device_ready()        (device_status &   0x02)

#define Host_set_ms_configured()      (device_status |=  0x04)
#define Host_clear_ms_configured()    (device_status &= ~0x04)
#define Is_host_ms_configured()       (device_status &   0x04)

#define Host_clear_device_status()    (device_status =   0x00)
//! @}

//! wSWAP
//! This macro swaps the uint8_t order in words.
//!
//! @param x        (uint16_t) the 16 bit word to swap
//!
//! @return         (uint16_t) the 16 bit word x with the 2 bytes swaped

#define wSWAP(x)        \
   (   (((x)>>8)&0x00FF) \
   |   (((x)<<8)&0xFF00) \
   )


//! Usb_write_word_enum_struc
//! This macro help to fill the uint16_t fill in USB enumeration struct.
//! Depending on the CPU architecture, the macro swap or not the nibbles
//!
//! @param x        (uint16_t) the 16 bit word to be written
//!
//! @return         (uint16_t) the 16 bit word written
#if !defined(BIG_ENDIAN) && !defined(LITTLE_ENDIAN)
	#error YOU MUST Define the Endian Type of target: LITTLE_ENDIAN or BIG_ENDIAN
#endif
#ifdef LITTLE_ENDIAN
	#define Usb_write_word_enum_struc(x)	(x)
#else //BIG_ENDIAN
	#define Usb_write_word_enum_struc(x)	(wSWAP(x))
#endif


//! @}

//_____ D E C L A R A T I O N ______________________________________________

uint8_t      usb_config_ep                (uint8_t, uint8_t);
uint8_t      usb_select_enpoint_interrupt (void);
uint16_t     usb_get_nb_byte_epw          (void);
uint8_t      usb_send_packet              (uint8_t , uint8_t*, uint8_t);
uint8_t      usb_read_packet              (uint8_t , uint8_t*, uint8_t);
void    usb_halt_endpoint            (uint8_t);
void    usb_reset_endpoint           (uint8_t);
uint8_t      usb_init_device              (void);

uint8_t      host_config_pipe             (uint8_t, uint8_t);
uint8_t      host_determine_pipe_size     (uint16_t);

#endif  // _USB_DRV_H_

