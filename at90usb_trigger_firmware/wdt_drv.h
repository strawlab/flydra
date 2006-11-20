//! @file $RCSfile: wdt_drv.h,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Use of this program is subject to Atmel's End User License Agreement.
//! Please read file license.txt for copyright notice.
//!
//! @brief This file contains the Watchdog low level driver definition
//!
//! @version $Revision: 1.6 $ $Name: at90usb128-demo-hidgen-last $ $Id: wdt_drv.h,v 1.6 2005/04/13 09:31:49 rletendu Exp $
//!
//! @todo
//! @bug


#ifndef _WDT_DRV_H_
#define _WDT_DRV_H_

//_____ I N C L U D E S ____________________________________________________




//_____ M A C R O S ________________________________________________________

#define Is_ext_reset()  ((MCUSR&(1<<EXTRF)) ? TRUE:FALSE)
#define Ack_ext_reset() (MCUSR= ~(1<<EXTRF))
#define Is_POR_reset()  ((MCUSR&(1<<(MCUSR= ~(1<<PORF)))) ? TRUE:FALSE)
#define Ack_POR_reset() (MCUSR= ~(1<<PORF))
#define Is_BOD_reset()  ((MCUSR&(1<<BORF)) ? TRUE:FALSE)
#define Ack_BOD_reset() (MCUSR= ~(1<<BORF))
#define Is_wdt_reset()  ((MCUSR&(1<<WDRF)) ? TRUE:FALSE)
#define Ack_wdt_reset() (MCUSR= ~(1<<WDRF))


#define Wdt_reset_instruction()	({asm volatile ("wdr"::);})
#define Wdt_clear_flag()			(Ack_wdt_reset())
#define Wdt_change_enable()		(WDTCSR |= (1<<WDCE) | (1<<WDE))
#define Wdt_enable_16ms()			(WDTCSR =  (1<<WDE))
#define Wdt_enable_32ms()			(WDTCSR =  (1<<WDE) | (1<<WDP0) )
#define Wdt_enable_64ms()			(WDTCSR =  (1<<WDE) | (1<<WDP1) )
#define Wdt_enable_125ms()			(WDTCSR =  (1<<WDE) | (1<<WDP1) | (1<<WDP0))
#define Wdt_enable_250ms()			(WDTCSR =  (1<<WDE) | (1<<WDP2) )
#define Wdt_enable_500ms()			(WDTCSR =  (1<<WDE) | (1<<WDP2) | (1<<WDP0))
#define Wdt_enable_1s()				(WDTCSR =  (1<<WDE) | (1<<WDP2) | (1<<WDP1))
#define Wdt_enable_2s()				(WDTCSR =  (1<<WDE) | (1<<WDP2) | (1<<WDP1) | (1<<WDP0))
#define Wdt_enable_4s()				(WDTCSR =  (1<<WDE) | (1<<WDP3) | (1<<WDP0))
#define Wdt_enable_8s()				(WDTCSR =  (1<<WDE) | (1<<WDP3) | (1<<WDP1))

#define Wdt_interrupt_16ms()		(WDTCSR =  (1<<WDIE))
#define Wdt_interrupt_32ms()		(WDTCSR =  (1<<WDIE) | (1<<WDP0) )
#define Wdt_interrupt_64ms()		(WDTCSR =  (1<<WDIE) | (1<<WDP1) )
#define Wdt_interrupt_125ms()		(WDTCSR =  (1<<WDIE) | (1<<WDP1) | (1<<WDP0))
#define Wdt_interrupt_250ms()		(WDTCSR =  (1<<WDIE) | (1<<WDP2) )
#define Wdt_interrupt_500ms()		(WDTCSR =  (1<<WDIE) | (1<<WDP2) | (1<<WDP0))
#define Wdt_interrupt_1s()			(WDTCSR =  (1<<WDIE) | (1<<WDP2) | (1<<WDP1))
#define Wdt_interrupt_2s()			(WDTCSR =  (1<<WDIE) | (1<<WDP2) | (1<<WDP1) | (1<<WDP0))
#define Wdt_interrupt_4s()			(WDTCSR =  (1<<WDIE) | (1<<WDP3) | (1<<WDP0))
#define Wdt_interrupt_8s()			(WDTCSR =  (1<<WDIE) | (1<<WDP3) | (1<<WDP1))

#define Wdt_enable_reserved5()	(WDTCSR =  (1<<WDE) | (1<<WDP3) | (1<<WDP2) | (1<<WDP1) | (1<<WDP0))
#define Wdt_stop()					(WDTCSR = 0x00)

#define Wdt_ack_interrupt()		(WDTCSR = ~(1<<WDIF))

//! Wdt_off.
//!
//! This macro stops the hardware watchdog timer.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_off()						(Wdt_reset_instruction(),	\
											 Wdt_clear_flag(),        	\
											 Wdt_change_enable(),     	\
											 Wdt_stop())




//! wdt_change_16ms.
//!
//! This macro activates the hardware watchdog timer for 16ms timeout.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_16ms()			(Wdt_reset_instruction(), \
											 Wdt_change_enable(),     \
											 Wdt_enable_32ms() )
										
//! wdt_change_32ms.
//!
//! This macro activates the hardware watchdog timer for 32ms timeout.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_32ms()			(Wdt_reset_instruction(), \
											 Wdt_change_enable(),     \
											 Wdt_enable_32ms() )


//! wdt_change_64ms.
//!
//! This macro activates the hardware watchdog timer for 64ms timeout.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_64ms()			(Wdt_reset_instruction(), \
											 Wdt_change_enable(),     \
											 Wdt_enable_64ms() )




//! wdt_change_32ms.
//!
//! This macro activates the hardware watchdog timer for 125ms timeout.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_125ms()			(Wdt_reset_instruction(), \
											 Wdt_change_enable(),     \
											 Wdt_enable_125ms() )

//! wdt_change_250ms.
//!
//! This macro activates the hardware watchdog timer for 250ms timeout.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_250ms()			(Wdt_reset_instruction(), \
											 Wdt_change_enable(),     \
											 Wdt_enable_250ms() )

//! wdt_change_500ms.
//!
//! This macro activates the hardware watchdog timer for 500ms timeout.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_500ms()			(Wdt_reset_instruction(), \
											 Wdt_change_enable(),     \
											 Wdt_enable_500ms() )

//! wdt_change_1s.
//!
//! This macro activates the hardware watchdog timer for 1s timeout.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_1s()				(Wdt_reset_instruction(), \
											 Wdt_change_enable(),     \
											 Wdt_enable_1s() )


//! wdt_change_2s.
//!
//! This macro activates the hardware watchdog timer for 2s timeout.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_2s()				(Wdt_reset_instruction(), \
											 Wdt_change_enable(),     \
											 Wdt_enable_2s() )
//! wdt_change_4s.
//!
//! This macro activates the hardware watchdog timer for 4s timeout.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_4s()				(Wdt_reset_instruction(), \
											 Wdt_change_enable(),     \
											 Wdt_enable_4s() )


//! wdt_change_8s.
//!
//! This macro activates the hardware watchdog timer for 8s timeout.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_8s()				(Wdt_reset_instruction(), \
											 Wdt_change_enable(),     \
											 Wdt_enable_8s() )


//! wdt_change_interrupt_16ms.
//!
//! This macro activates the hardware watchdog timer for 16ms interrupt.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_interrupt_16ms()		(Wdt_reset_instruction(), \
													 Wdt_change_enable(),     \
													 Wdt_interrupt_16ms() )

//! wdt_change_interrupt_32ms.
//!
//! This macro activates the hardware watchdog timer for 32ms interrupt.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_interrupt_32ms()		(Wdt_reset_instruction(), \
													 Wdt_change_enable(),     \
													 Wdt_interrupt_32ms() )

//! wdt_change_interrupt_64ms.
//!
//! This macro activates the hardware watchdog timer for 64ms interrupt.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_interrupt_64ms()		(Wdt_reset_instruction(), \
													 Wdt_change_enable(),     \
													 Wdt_interrupt_64ms() )

//! wdt_change_interrupt_125ms.
//!
//! This macro activates the hardware watchdog timer for 125ms interrupt.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_interrupt_125ms()		(Wdt_reset_instruction(), \
								    		      	 Wdt_change_enable(),     \
														 Wdt_interrupt_125ms() )

//! wdt_change_interrupt_250ms.
//!
//! This macro activates the hardware watchdog timer for 250ms interrupt.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_interrupt_250ms()		(Wdt_reset_instruction(), \
						Wdt_change_enable(),     \
						Wdt_interrupt_250ms() )

//! wdt_change_interrupt_500ms.
//!
//! This macro activates the hardware watchdog timer for 500ms interrupt.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_interrupt_500ms()		(Wdt_reset_instruction(), \
						Wdt_change_enable(),     \
						Wdt_interrupt_500ms() )

//! wdt_change_interrupt_1s.
//!
//! This macro activates the hardware watchdog timer for 1s interrupt.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_interrupt_1s()		(Wdt_reset_instruction(), \
						Wdt_change_enable(),     \
						Wdt_interrupt_1s() )

//! wdt_change_interrupt_2s.
//!
//! This macro activates the hardware watchdog timer for 2s interrupt.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_interrupt_2s()		(Wdt_reset_instruction(), \
						Wdt_change_enable(),     \
						Wdt_interrupt_2s() )

//! wdt_change_interrupt_4s.
//!
//! This macro activates the hardware watchdog timer for 4s interrupt.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_interrupt_4s()		(Wdt_reset_instruction(), \
						Wdt_change_enable(),     \
						Wdt_interrupt_4s() )

//! wdt_change_interrupt_8s.
//!
//! This macro activates the hardware watchdog timer for 8s interrupt.
//!
//! @warning Interrupts should be disable before call to ensure
//! no timed sequence break.
//!
//! @param none
//!
//! @return none.
//!
#define Wdt_change_interrupt_8s()		(Wdt_reset_instruction(), \
						Wdt_change_enable(),     \
						Wdt_interrupt_8s() )

#define Wdt_change_reserved5()	(Wdt_reset_instruction(), \
											Wdt_change_enable(),     \
											Wdt_enable_reserved5() )

#define Soft_reset()					{asm("jmp 0000");}

//_____ D E C L A R A T I O N ______________________________________________





#endif  // _WDT_DRV_H_

