/****************************************************************************
 * 
 *                              serial_comm.c
 *
 ****************************************************************************
 *
 * Serial communications functions. 
 *
 ****************************************************************************
 *
 * Revision History: 
 *
 *		(11/07/2003) Original version - William Dickson
 *		(7/13/04) revised for Linux - John Bender
 *
 ****************************************************************************/ 

#include <stdlib.h>		  /* Standard Library								*/
#include <stdio.h>		  /* Standard I/O									*/
#include "serial_comm.h"  /* Serial communications header file              */ 

#include <errno.h>

const char *SC_INFO_STR_RC[SC_NUM_RC] =	 /* Error code information strings  */
{
	"SC_SUCCESS_RC: succesful function call", 
	"SC_OPEN_ERROR_RC: unable to open port", 
	"SC_CONFIG_ERROR_RC: unable to configure port", 
	"SC_PURGE_ERROR_RC: unable to purge port before closing",
	"SC_CLOSE_ERROR_RC: unable to close port", 
	"SC_WRITE_ERROR_RC: unable to write data to port", 
	"SC_WRITE_SHORT_RC: less data written to port than expected",
	"SC_READ_ERROR_RC: unable to read data from port", 
};

/****************************************************************************
 * 
 * Function: sc_open_port
 *
 ****************************************************************************
 *
 * Opens communication port using CreateFile, and sets port 
 * properties. Returns zero on success returns non zero if failure.
 *
 ****************************************************************************/

long sc_open_port(
  int *hcomm,      /* Handle for communication port*/
  char *port_name  /* Communications port, e.g. "/dev/ttyS0" */
)
{
	/* Variable Declarations ************************************************/
	struct termios tio;
	/* End Variable Declarations ********************************************/

	*hcomm = open( port_name, O_RDWR | O_NOCTTY | O_NONBLOCK | O_ASYNC );
	if( *hcomm <= 0 ) return SC_OPEN_ERROR_RC;

	tio.c_iflag = 0;
	tio.c_oflag = OPOST;
	tio.c_cflag = SC_BAUD_RATE | CS8 | CLOCAL;
	tio.c_lflag = 0;
	if( tcsetattr( *hcomm, TCSANOW, &tio ) != 0 )
		return SC_CONFIG_ERROR_RC;

	return SC_SUCCESS_RC;
}

/* End sc_open_port *********************************************************/


/****************************************************************************
 *
 * Function: sc_close_port
 *
 ****************************************************************************
 *
 * Purges and closes communications port.
 *
 ****************************************************************************/

long sc_close_port(
  int *hcomm    /* Handle for communications port */
)
{
	/* Variable Declarations ************************************************/
	/* End Variable Declarations ********************************************/

	if( close( *hcomm ) != 0 )
		return SC_CLOSE_ERROR_RC;

	hcomm = 0;

	return SC_SUCCESS_RC;
}

/* End sc_close_port ********************************************************/


/****************************************************************************
 * 
 * Function: sc_send_cmd
 *
 ****************************************************************************
 *
 * Sends a command to the communications port
 *
 ****************************************************************************/

long sc_send_cmd(
  int *hcomm,   /* Handle for communications port */
  char *cmd_buffer,  /* Command  string to send */         
  int cmd_buffer_len /*Length of command to send  */
)
{
	/* Variable Declarations ***********************************************/
	size_t bytes_out; /* Number of bytes to write	*/
	size_t bytes_out_act;	/* Number of bytes actually written*/
#if 1
int i,j;
	/* End Variable Declarations ********************************************/
printf( "  sending serial\t" );
for( j = 0; j < cmd_buffer_len; j++ )
{
  for( i = 7; i >= 0; i-- )
    printf( "%d", (cmd_buffer[j] >> i) & 1 );
  printf( " " );
}
printf( "\n" );
#endif

	bytes_out = cmd_buffer_len;

	bytes_out_act = write( *hcomm, cmd_buffer, bytes_out );

	if( bytes_out_act == -1 ) return SC_WRITE_ERROR_RC;
	else if( bytes_out_act != bytes_out ) return SC_WRITE_SHORT_RC;
	return SC_SUCCESS_RC;
}

/* End sc_send_cmd **********************************************************/


/****************************************************************************
 * 
 * Function: sc_receive_msg 
 * 
 ****************************************************************************
 *
 * Reads message from port.
 *
 ****************************************************************************/

long sc_receive_msg(
  int *hcomm,          /* HANDLE for communications port*/
  char *msg_buffer,     /* Message      read    */
  int *msg_size         /* Size of message read, init to max size to read */
)
{
	/* Variable Declarations ************************************************/
	size_t bytes_read;/* Number of bytes read */
	/* End Variable Declarations ********************************************/

	/* Read from port */
	bytes_read = read( *hcomm, msg_buffer, msg_size );

	if( bytes_read < 0 ) return SC_READ_ERROR_RC;

	*msg_size = (int)bytes_read;

	return SC_SUCCESS_RC;
}

/* End sc_receive_msg *******************************************************/


/****************************************************************************
 *
 * Function: sc_get_error_index
 *
 ****************************************************************************
 *
 * Get indices of error messages associated with a given return code
 *
 ****************************************************************************/

void sc_get_error_index(
  int return_code,      /* return code  */
  int *indices,         /* Indices of errors in return code*/
  int *num_indices    /* Number of error indices returned*/
)
{
	/* Varibale Declarations ************************************************/

	int i;				/* Index											*/
	long mask = 1;		/* Mask for extracting error codes					*/

	/* End Variable Declarations ********************************************/

	if ( !return_code ) {
		*num_indices = 1;
		indices[0] = 0;
	} else {
		*num_indices = 0;
	
		for ( i = 1; i < SC_NUM_RC; i++ ) {
			if ( mask & return_code ) {
				*num_indices += 1;
				indices[*num_indices - 1] = i;
			}
		
			mask *= 2;
		}
	}

	return;
}
/* End sc_get_error_index ***************************************************/


/****************************************************************************
 *
 * Function: sc_print_error_msg
 *
 ****************************************************************************
 *
 * Print error messages associated with the return code, if flag
 * is true. 
 *
 ****************************************************************************/

void sc_print_error_msg(
  int return_code,      /* Return code*/
  int flag              /* If flag != 0 then print messages*/
)
{
	/* Variable Declarations ************************************************/
	int i;					/* Index										*/
	int indices[SC_NUM_RC];	/* Indices of return codes						*/
	int num_indices;		/* Number of return code indices				*/
	/* End Variable Declarations ********************************************/

	if( flag ) {
		sc_get_error_index( return_code, indices, &num_indices );
		printf( "SCReturn code ---------------------------------- \n" );

		for( i = 0; i < num_indices; i++ ) {
			printf( SC_INFO_STR_RC[ indices[i] ] );
			printf( "\n" );
		}
		printf( "----------------------------------------------- \n" );
	}

	return;
}

/* End sc_print_error_msg ***************************************************/

