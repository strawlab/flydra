/*
  Hardware Serial Routines for ATMega128
  By Robert Bailey
	
  Revision History:
  09.05.02  RB  Created
  09.15.02  RB  Added UART_SendHex
  09.17.02  RB  Added UART_GetHex
  07.08.04  RB  Cleaned up and consolidated UART and USART routines
  07.08.04  RB  Branched this code to work with the dual USARTs of the mega128
*/

#include "ser128.h"

#include <avr/io.h>
#include <avr/signal.h>
#include <avr/pgmspace.h>
#include <avr/interrupt.h>


/* UART Routine Variables */
volatile unsigned int uart_tx_buf_cnt0;   /* number of buffer slots used */
volatile unsigned int uart_rx_buf_cnt0;
unsigned char *uart_tx_tail_ptr0, *uart_tx_head_ptr0; /* buffer pointers */
unsigned char *uart_rx_tail_ptr0, *uart_rx_head_ptr0;
unsigned char uart_tx_buffer0[UART_BUF_SIZE0];        /* buffers */
unsigned char uart_rx_buffer0[UART_BUF_SIZE0];

volatile unsigned int uart_tx_buf_cnt1;   /* number of buffer slots used */
volatile unsigned int uart_rx_buf_cnt1;
unsigned char *uart_tx_tail_ptr1, *uart_tx_head_ptr1; /* buffer pointers */
unsigned char *uart_rx_tail_ptr1, *uart_rx_head_ptr1;
unsigned char uart_tx_buffer1[UART_BUF_SIZE1];        /* buffers */
unsigned char uart_rx_buffer1[UART_BUF_SIZE1];

/* UART Routines */

/*
Function Name: UART_Init

Description: Initializes the UART

Arguments: uart_sel - select which uart to initialize

Return Values: none

Limitations: Cannot handle chips with Dual USARTS
Notes:
*/
void UART_Init(unsigned char uart_sel)
{
  /* disable interrupts while we setup the uart */
  cli();
  if(uart_sel==0)
  {
    /* set baud rate */
    outp(0x00,UBRR0H);
    outp((unsigned char)UART_BAUD_SELECT0,UBRR0L);
  
    /* setup USART0 */
    outp((0<<UMSEL)|(0<<UPM1)|(0<<UPM0)|(0<<USBS)|(1<<UCSZ1)|(1<<UCSZ0),UCSR0C);
    outp(0x00,UCSR0A);
  
    /* enable RxD/TxD, rxc IRQ */
    outp((1<<RXCIE)|(1<<RXEN)|(1<<TXEN),UCSR0B);  

    /* set-up buffers */
    uart_tx_tail_ptr0  = uart_tx_head_ptr0 = uart_tx_buffer0;
    uart_rx_tail_ptr0  = uart_rx_head_ptr0 = uart_rx_buffer0;
    uart_tx_buf_cnt0 = 0;
    uart_rx_buf_cnt0 = 0;
  }
  
  if(uart_sel==1)
  {
    /* set baud rate */
    outp(0x00,UBRR1H);
    outp((unsigned char)UART_BAUD_SELECT1,UBRR1L);
  
    /* setup USART1 */
    outp((0<<UMSEL)|(0<<UPM1)|(0<<UPM0)|(0<<USBS)|(1<<UCSZ1)|(1<<UCSZ0),UCSR1C);
    outp(0x00,UCSR1A);
  
    /* enable RxD/TxD, rxc IRQ */
    outp((1<<RXCIE)|(1<<RXEN)|(1<<TXEN),UCSR1B);  

    /* set-up buffers */
    uart_tx_tail_ptr1  = uart_tx_head_ptr1 = uart_tx_buffer1;
    uart_rx_tail_ptr1  = uart_rx_head_ptr1 = uart_rx_buffer1;
    uart_tx_buf_cnt1 = 0;
    uart_rx_buf_cnt1 = 0;
  }
  
  /* enable interrupts */
  sei();
}

/* UART Data Register Empty IRQ handler */
SIGNAL(SIG_UART0_DATA)
{
  if (uart_tx_buf_cnt0 > 0) {
    outp(*uart_tx_head_ptr0, UDR0);                     /* write byte out */
    if(++uart_tx_head_ptr0 >= uart_tx_buffer0+UART_BUF_SIZE0) /* Wrap ptr */
      uart_tx_head_ptr0 = uart_tx_buffer0;
    if(--uart_tx_buf_cnt0 == 0)                     /* if buffer is empty */
      cbi(UCSR0B,UDRIE);                                 /* disable UDRIE */
  }  
}

SIGNAL(SIG_UART1_DATA)
{
  if (uart_tx_buf_cnt1 > 0) {
    outp(*uart_tx_head_ptr1, UDR1);                     /* write byte out */
    if(++uart_tx_head_ptr1 >= uart_tx_buffer1+UART_BUF_SIZE1) /* Wrap ptr */
      uart_tx_head_ptr1 = uart_tx_buffer1;
    if(--uart_tx_buf_cnt1 == 0)                     /* if buffer is empty */
      cbi(UCSR1B,UDRIE);                                 /* disable UDRIE */
  }  
}

/* UART Receive IRQ */
SIGNAL(SIG_UART0_RECV)
{
  *uart_rx_tail_ptr0=inp(UDR0);                 /* read byte from recieve reg */
  uart_rx_buf_cnt0++;
  if(++uart_rx_tail_ptr0 >= uart_rx_buffer0+UART_BUF_SIZE0) /*Wrap the pointer*/
    uart_rx_tail_ptr0=uart_rx_buffer0;
}

SIGNAL(SIG_UART1_RECV)
{
  *uart_rx_tail_ptr1=inp(UDR1);                 /* read byte from recieve reg */
  uart_rx_buf_cnt1++;
  if(++uart_rx_tail_ptr1 >= uart_rx_buffer1+UART_BUF_SIZE1) /*Wrap the pointer*/
    uart_rx_tail_ptr1=uart_rx_buffer1;
}

/*
Function Name: UART_Putchar

Description: sends one char, c, to the serial port

Arguments:  unsigned char c = char to send
            unsigned char uart_sel = which uart to tx on

Return Values: none

Limitations: 
Notes:
*/
void UART_Putchar(unsigned char uart_sel,unsigned char c)
{
  if(uart_sel==0)
  {
    while(uart_tx_buf_cnt0>=UART_BUF_SIZE0);
    cli();
    uart_tx_buf_cnt0++;
    *uart_tx_tail_ptr0=c;                           /* store char in buffer */
    if(++uart_tx_tail_ptr0 >= uart_tx_buffer0 + UART_BUF_SIZE0) /* Wrap ptr */
      uart_tx_tail_ptr0=uart_tx_buffer0;
    sbi(UCSR0B,UDRIE);                            /* make sure UDRIE is set */
  }
  
  if(uart_sel==1)
  {
    while(uart_tx_buf_cnt1>=UART_BUF_SIZE1);
    cli();
    uart_tx_buf_cnt1++;
    *uart_tx_tail_ptr1=c;                           /* store char in buffer */
    if(++uart_tx_tail_ptr1 >= uart_tx_buffer1 + UART_BUF_SIZE1) /* Wrap ptr */
      uart_tx_tail_ptr1=uart_tx_buffer1;
    sbi(UCSR1B,UDRIE);                            /* make sure UDRIE is set */
  }
  
  sei();
}

/*
Function Name: UART_Getchar

Description: Gets one char from the serial port.

Arguments: unsigned char uart_sel = which uart to rx from

Return Values: unsigned char c = rxd char

Limitations: 
Notes:
*/
unsigned char UART_Getchar(unsigned char uart_sel)
{
  unsigned char c;
  c=0;  /* initialize c */
  if(uart_sel==0)
  {
    while(uart_rx_buf_cnt0==0);
  
    cli();
    uart_rx_buf_cnt0--;
    c=*uart_rx_head_ptr0;                         /* get char from buffer */
    if(++uart_rx_head_ptr0 >= uart_rx_buffer0+UART_BUF_SIZE0) /* Wrap ptr */
      uart_rx_head_ptr0 = uart_rx_buffer0;
  }
  if(uart_sel==1)
  {
    while(uart_rx_buf_cnt1==0);
  
    cli();
    uart_rx_buf_cnt1--;
    c=*uart_rx_head_ptr1;                         /* get char from buffer */
    if(++uart_rx_head_ptr1 >= uart_rx_buffer1+UART_BUF_SIZE1) /* Wrap ptr */
      uart_rx_head_ptr1 = uart_rx_buffer1;
  }
  sei();
  return c;
}

/*
Function Name: UART_CharReady

Description: returns 0 if there is nothing in the uart rx buffer, 
              returns >0 if a char is available.

Arguments: unsigned char uart_sel

Return Values: unsigned int

Limitations: 
Notes:
*/
unsigned int UART_CharReady(unsigned char uart_sel)
{
  if(uart_sel==0)
  {
    return uart_rx_buf_cnt0;
  }
  else
  {
    return uart_rx_buf_cnt1;
  }
}

/*
Function Name: UART_Putstr

Description: Sends a string in program memory to the serial port

Arguments:  unsigned char *s = ptr to the string to send
            unsigned char uart_sel

Return Values: none

Limitations: 
Notes:
*/
void UART_Putstr(unsigned char uart_sel,unsigned char *s)
{
  while(PRG_RDB(s)) {
    UART_Putchar(uart_sel,PRG_RDB(s));
    s++;
  }
}

/* Sends a Carriage Return and Line Feed to the serial port */
void UART_CRLF(unsigned char uart_sel)
{
  UART_Putchar(uart_sel,CR);
  UART_Putchar(uart_sel,LF);
}

/*
Function Name: UART_SendHex

Description: sends an 8-bit hex value to the serial port in character representation

Arguments:  unsigned char c = hex value
            unsigned char uart_sel

Return Values: none

Limitations: 
Notes:
*/
void UART_SendHex(unsigned char uart_sel,unsigned char c)
{
  unsigned char hex1;
  unsigned char hex2;
  hex1 = c / 0x10;
  hex2 = c - (hex1 * 0x10);
  if(hex1>=10)
  {
    hex1 = hex1 + ('A'-10);
  }
  else
  {
    hex1 = hex1 + '0';
  }
  
  if(hex2>=10)
  {
    hex2 = hex2 + ('A'-10);
  }
  else
  {
    hex2 = hex2 + '0';
  }
	
  UART_Putchar(uart_sel,hex1);
  UART_Putchar(uart_sel,hex2);
}

/*
Function Name: UART_GetHex

Description: gets a 16-bit hex value from the serial port

Arguments: unsigned char uart_sel

Return Values: long hex = the rxd hex value

Limitations: 
Notes:
*/
long UART_GetHex(unsigned char uart_sel)
{
  unsigned char rxdstr[8];
  long hex;
  unsigned char index,j;
  hex = 0;
  
  for(index=0;index<8;index++)
    rxdstr[index] = 0;
	
  index = 0;
  
  do
  {
    while(!(UART_CharReady(uart_sel)));
    rxdstr[index] = UART_Getchar(uart_sel);
    UART_Putchar(uart_sel,rxdstr[index]);
    index++;
    
  }while((rxdstr[index-1]!=CR)&(index<8));
  
  if(rxdstr[index-1]==CR)
    index--;
	
  for(j=0;j<index;j++)
  {
    if(rxdstr[j]>='A')
    {
      rxdstr[j] = rxdstr[j] - ('A'-10);
    }
    else
    {
      rxdstr[j] = rxdstr[j] - '0';
    }
    hex = hex * 0x10;
    hex = hex + rxdstr[j];
  }
  return hex;
}

void UART_SendNum(unsigned char uart_sel,unsigned char c)
{
  unsigned char num1;
  unsigned char num2;
  unsigned char num3;
  num1 = c / 100;
  num2 = (c%100) / 10;
  num3 = (c%10);
  
  num1 += '0';
  num2 += '0';
  num3 += '0';
  
  UART_Putchar(uart_sel,num1);
  UART_Putchar(uart_sel,num2);
  UART_Putchar(uart_sel,num3);
}
