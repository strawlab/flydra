//! @file $RCSfile: conf_scheduler.h,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Please read file license.txt for copyright notice.
//!
//! This file contains the scheduler configuration definition
//!
//! @version $Revision: 1.1 $ $Name: at90usb128-demo-hidgen-last $ $Id: conf_scheduler.h,v 1.1 2005/11/16 18:25:20 rletendu Exp $
//!
//! @todo
//! @bug

#ifndef _CONF_SCHEDULER_H_
#define _CONF_SCHEDULER_H_



/*--------------- SCHEDULER CONFIGURATION --------------*/
#define SCHEDULER_TYPE          SCHEDULER_FREE  // SCHEDULER_(TIMED|TASK|FREE|CUSTOM)
#define Scheduler_task_1_init   usb_task_init
#define Scheduler_task_1        usb_task
#define Scheduler_task_2_init   trigger_task_init
#define Scheduler_task_2        trigger_task
/*
#define Scheduler_task_3_init   framecount_task_init
#define Scheduler_task_3        framecount_task
*/

#endif  //! _CONF_SCHEDULER_H_

