/*
  USART Routines Header File for ATMega128
  By Robert Bailey
*/

/*
Description: Header file for the USART routines. Defines constants and 
contains prototypes for the USART routines.
*/

/* USART Definitions */
#define F_CPU 8000000               /* 8.00 mhz */
#define UART_BUF_SIZE0 16           /* size of Tx buffer */
#define UART_BAUD_RATE0 19200
  /* automatically calcuate baud register value */
#define UART_BAUD_SELECT0 (F_CPU/(UART_BAUD_RATE0*16l)-1)
/*#define UART_BAUD_SELECT0 0x19*/

#define UART_BUF_SIZE1 16           /* size of Tx buffer */
#define UART_BAUD_RATE1 19200
  /* automatically calcuate baud register value */
#define UART_BAUD_SELECT1 (F_CPU/(UART_BAUD_RATE1*16l)-1)
/*#define UART_BAUD_SELECT1 0x19*/

#define CTRLC 3
#define CR 13
#define LF 10
#define BKSPC 8

/* USART Routine Prototypes */
void UART_Init(unsigned char uart_sel);
void UART_Putchar(unsigned char uart_sel,unsigned char c);
unsigned char UART_Getchar(unsigned char uart_sel);
void UART_Putstr(unsigned char uart_sel,unsigned char *s);
void UART_CRLF(unsigned char uart_sel);
unsigned int UART_CharReady(unsigned char uart_sel);
void UART_SendHex(unsigned char uart_sel,unsigned char c);
long UART_GetHex(unsigned char uart_sel);
void UART_SendNum(unsigned char uart_sel,unsigned char c);
