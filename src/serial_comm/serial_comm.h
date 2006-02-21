/***************************************************************************
 * 
 *                            serial_comm.h 
 *
 ***************************************************************************
 *
 * Header file for serial port communications (serial_comm.c).  
 *
 * Function prototypes: open_port, close_port, send_cmd, receive_msg, 
 * get_error_index, print_error_msg. 
 *
 *
 ***************************************************************************
 *
 *  Revision History: 
 *
 *		(11/07/2003) Original version - William Dickson 
 *		(7/13/04) revised for Linux - John Bender
 *		(10/11/04) changed baud rate from B19200 to B57600 - JAB
 *		(3/2/05) changed baud rate back
 *
 ****************************************************************************/


#ifndef SERIAL_COMM_H
#define SERIAL_COMM_H 

#include <fcntl.h>
#include <termios.h>

/* Global Variables and Constants *******************************************/  

#define SC_COMM_PORT "/dev/ttyS0"	/* Communications Port	*/
#define SC_BAUD_RATE B19200		/* Baud rate	*/ 
/*#define SC_BAUD_RATE B57600*/		/* Baud rate	*/ 

#define SC_CMD_BUF_SIZE 100			/* Size of cmd buffer	(send)			*/
#define SC_MSG_BUF_SIZE 100			/* Size of msg buffer	(recieve)		*/


/* Error Codes **************************************************************/

#define SC_SUCCESS_RC 0		/* Successful function completion */
#define SC_OPEN_ERROR_RC 1	/* Unable to open port */
#define SC_CONFIG_ERROR_RC 2	/* Unable to configure port */
#define SC_PURGE_ERROR_RC 4	/* Unable to purge port	*/
#define SC_CLOSE_ERROR_RC 8	/* Unable to close port	*/
#define SC_WRITE_ERROR_RC 16	/* Unable to write to port*/
#define SC_WRITE_SHORT_RC 32	/* Wrote less data to port than expected */
#define SC_READ_ERROR_RC 64	/* Unable to read from port*/ 

#define SC_NUM_RC 8/* Number of error return codes */
#define SC_REPORT_FLAG 1 /* If != 0 return codes are printed on an error */


/* Function Prototypes ******************************************************/

/* Open comm port */
long sc_open_port( 
  int *hcomm,	/* Handle for communication port*/
  char *port_name/* Communications port, e.g. "/dev/ttyS0"*/
);	

/* Close comm port	*/
long sc_close_port(  
  int *hcomm	/* Handle for communications*/
);				
	
/* Send command to communications port	*/
long sc_send_cmd(
  int *hcomm,		/* Handle for communications port*/
  char *cmd_buffer,	/* Command	string to send	*/ 
  int cmd_buffer_len	/* Length of command to send */
);

/* Receive message from communications port	*/ 
long sc_receive_msg( 
  int *hcomm,		/* HANDLE for communications port*/
  char *msg_buffer,	/* Message	read	*/
  int *msg_size		/* Size of message read	*/
);

/* Get error indices */
void sc_get_error_index(
  int return_code,	/* return code	*/
  int *indices,		/* Indices of errors in return code*/
  int *num_indices    /* Number of error indices returned*/
);

/* Print error messages */
void sc_print_error_msg(
  int return_code,	/* Return code*/
  int flag		/* If flag != 0 then print messages*/
);

#endif 
