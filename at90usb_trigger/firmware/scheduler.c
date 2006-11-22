//! @file $RCSfile: scheduler.c,v $
//!
//! Copyright (c) 2004 Atmel.
//!
//! Please read file license.txt for copyright notice.
//! @brief This file contains the scheduler routines
//!
//!Configuration:
//!  - SCHEDULER_TYPE in scheduler.h header file
//!  - Task & init for at least task number 1 must be defined
//!
//! @version $Revision: 1.11 $ $Name: at90usb128-demo-hidgen-last $ $Id: scheduler.c,v 1.11 2005/03/10 08:30:50 lguilhau Exp $
//!
//! @todo
//! @bug

//!_____ I N C L U D E S ____________________________________________________
#define _SCHEDULER_C_
#include "config.h"                         // system definition 
#include "conf_scheduler.h"                 // Configuration for the scheduler
#include "scheduler.h"                      // scheduler definition 


//!_____ M A C R O S ________________________________________________________
//!_____ D E F I N I T I O N ________________________________________________
#if SCHEDULER_TYPE != SCHEDULER_FREE
//! When SCHEDULER_TYPE != SCHEDULER_FREE, this flag control task calls.
bit   scheduler_tick_flag;
#endif

#ifdef TOKEN_MODE
//! Can be used to avoid that some tasks executes at same time. 
//! The tasks check if the token is free before executing. 
//! If the token is free, the tasks reserve it at the begin of the execution 
//! and release it at the end of the execution to enable next waiting tasks to 
//! run.	
Uchar token;
#endif

//!_____ D E C L A R A T I O N ______________________________________________
//! Scheduler initialization
//!
//! Task_x_init() and Task_x_fct() are defined in config.h
//!
//! @warning Code:XX bytes (function code length)
//!
//! @param  :none
//! @return :none
void scheduler_init (void)
{
   #ifdef Scheduler_time_init
      Scheduler_time_init();
   #endif
   #ifdef TOKEN_MODE
      token =  TOKEN_FREE;
   #endif
   #ifdef Scheduler_task_1_init
      Scheduler_task_1_init();  
      Scheduler_call_next_init();
   #endif
   #ifdef Scheduler_task_2_init
      Scheduler_task_2_init();  
      Scheduler_call_next_init();
   #endif
   #ifdef Scheduler_task_3_init
      Scheduler_task_3_init();  
      Scheduler_call_next_init();
   #endif
   #ifdef Scheduler_task_4_init
      Scheduler_task_4_init();  
      Scheduler_call_next_init();
   #endif
   #ifdef Scheduler_task_5_init
      Scheduler_task_5_init();  
      Scheduler_call_next_init();
   #endif
   #ifdef Scheduler_task_6_init
      Scheduler_task_6_init();  
      Scheduler_call_next_init();
   #endif
   #ifdef Scheduler_task_7_init
      Scheduler_task_7_init();  
      Scheduler_call_next_init();
   #endif
   #ifdef Scheduler_task_8_init
      Scheduler_task_8_init();  
      Scheduler_call_next_init();
   #endif
   #ifdef Scheduler_task_9_init
      Scheduler_task_9_init();  
      Scheduler_call_next_init();
   #endif
   #ifdef Scheduler_task_10_init
      Scheduler_task_10_init();
      Scheduler_call_next_init();
   #endif
   #ifdef Scheduler_task_11_init
      Scheduler_task_11_init();
      Scheduler_call_next_init();
   #endif
   Scheduler_reset_tick_flag();
}

//! Task execution scheduler
//!
//! @warning Code:XX bytes (function code length)
//!
//! @param  :none
//! @return :none
void scheduler_tasks (void)
{
   // To avoid uncalled segment warning if the empty function is not used
   scheduler_empty_fct();

   for(;;)
   {
      Scheduler_new_schedule();
      #ifdef Scheduler_task_1
         Scheduler_task_1();
         Scheduler_call_next_task();
      #endif
      #ifdef Scheduler_task_2
         Scheduler_task_2();
         Scheduler_call_next_task();
      #endif
      #ifdef Scheduler_task_3
         Scheduler_task_3();
         Scheduler_call_next_task();
      #endif
      #ifdef Scheduler_task_4
         Scheduler_task_4();
         Scheduler_call_next_task();
      #endif
      #ifdef Scheduler_task_5
         Scheduler_task_5();
         Scheduler_call_next_task();
      #endif
      #ifdef Scheduler_task_6
         Scheduler_task_6();
         Scheduler_call_next_task();
      #endif
      #ifdef Scheduler_task_7
         Scheduler_task_7();
         Scheduler_call_next_task();
      #endif
      #ifdef Scheduler_task_8
         Scheduler_task_8();
         Scheduler_call_next_task();
      #endif
      #ifdef Scheduler_task_9
         Scheduler_task_9();
         Scheduler_call_next_task();
      #endif
      #ifdef Scheduler_task_10
         Scheduler_task_10();
         Scheduler_call_next_task();
      #endif
      #ifdef Scheduler_task_11
         Scheduler_task_11();
         Scheduler_call_next_task();
      #endif
   }
}

//! Init & run the scheduler
//!
//! @warning Code:XX bytes (function code length)
//!
//! @param  :none
//! @return :none
void scheduler (void)
{
   scheduler_init();
   scheduler_tasks();
}


//! Do nothing
//! Avoid uncalled segment warning if the empty function is not used
//!
//! @warning Code:XX bytes (function code length)
//!
//! @param  :none
//! @return :none
void scheduler_empty_fct (void)
{
}

