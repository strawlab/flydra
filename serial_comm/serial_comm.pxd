#emacs, this is -*-Python-*- mode

cdef extern from "serial_comm.h":
    ctypedef enum SERIAL_COMM_DEFINES:
#        SC_COMM_PORT
        SC_BAUD_RATE
        SC_CMD_BUF_SIZE
        SC_MSG_BUF_SIZE

        SC_SUCCESS_RC
        SC_OPEN_ERROR_RC
        SC_CONFIG_ERROR_RC
        SC_PURGE_ERROR_RC
        SC_CLOSE_ERROR_RC
        SC_WRITE_ERROR_RC
        SC_WRITE_SHORT_RC
        SC_READ_ERROR_RC
        
        SC_NUM_RC
        SC_REPORT_FLAG

    long sc_open_port( int *hcomm, char *port_name)
    long sc_close_port( int *hcomm )
    long sc_send_cmd(int *hcomm, char *cmd_buffer, int cmd_buffer_len)
    long sc_receive_msg( int *hcomm, char *msg_buffer, int *msg_size )
    void sc_get_error_index( int return_code, int *indices, int *num_indices)
    void sc_print_error_msg( int return_code, int flag )
