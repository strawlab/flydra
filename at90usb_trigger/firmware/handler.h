/*
    Interrupt Handler Routines Header File
    By Robert Bailey

    Revision History:
    11.15.02   RB   Created
    06.26.03   RB   Cleaned up
*/

/*
Description: Header file for the interrupt handler routines. Defines constants
and contains prototypes for the Interrupt Handler routines.
*/

/* Interrupt Handler Definitions */
#define HANDLER_MAX 4              /* number of functions handler can handle */

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

/* Interrupt Handler Routine Prototypes */
void Handler_Init(void);
void Reg_Handler(void* fptr,
		 unsigned long s_cnt,
		 unsigned char priority,
		 unsigned char msk);
