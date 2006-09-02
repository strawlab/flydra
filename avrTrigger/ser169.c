#include <avr/io.h>
#include <avr/signal.h>
#include <avr/interrupt.h>

#define sbiBF(port,bit)  (port |= (1<<bit))   //set bit in port
#define cbiBF(port,bit)  (port &= ~(1<<bit))  //clear bit in port

#define UART_BUF_SIZE 16           /* size of Tx buffer */

/* UART Routine Variables */
volatile unsigned int uart_tx_buf_cnt;   /* number of buffer slots used */
volatile unsigned int uart_rx_buf_cnt;
unsigned char *uart_tx_tail_ptr, *uart_tx_head_ptr; /* buffer pointers */
unsigned char *uart_rx_tail_ptr, *uart_rx_head_ptr;
unsigned char uart_tx_buffer[UART_BUF_SIZE];        /* buffers */
unsigned char uart_rx_buffer[UART_BUF_SIZE];

void UART_init(void) {
  unsigned int ubrr;

  cli();

  // Set baud rate
  ubrr = 12; // 4800 bps, see atmega169 manual pg. 174
  UBRRH = (unsigned char)(ubrr>>8);
  UBRRL = (unsigned char)ubrr;

  //  outp((0<<UMSEL)|(0<<UPM1)|(0<<UPM0)|(0<<USBS)|(1<<UCSZ1)|(1<<UCSZ0),UCSR0C);
  //  outp(0x00,UCSR0A);
  //  UCSRC = (0<<UMSEL)|(0<<UPM1)|(0<<UPM0)|(0<<USBS)|(1<<UCSZ1)|(1<<UCSZ0);
  UCSRA = 0x00;

  // Enable receiver, transmitter, and interrupts
  UCSRB = (1<<RXEN) | (1<<TXEN) | (1<<RXCIE);
  //UCSRB = (1<<RXEN) | (1<<TXEN);
  
  // Set frame format 8N1
  // UCSRC defaults to 8N1 = (3<<UCSZ0)

  // Set frame format 7E1
  //UCSRC = (1<<UPM1)|(1<<UCSZ1);

  /* set-up buffers */
  uart_tx_tail_ptr  = uart_tx_head_ptr = uart_tx_buffer;
  uart_rx_tail_ptr  = uart_rx_head_ptr = uart_rx_buffer;
  uart_tx_buf_cnt = 0;
  uart_rx_buf_cnt = 0;

  sei();
}

/* UART Data Register Empty IRQ handler */
SIGNAL(SIG_USART_DATA)
{
  if (uart_tx_buf_cnt > 0) {
    UDR = *uart_tx_head_ptr; /* write byte out */
//    outp(*uart_tx_head_ptr, UDR);                     

    if(++uart_tx_head_ptr >= uart_tx_buffer+UART_BUF_SIZE) /* Wrap ptr */
      uart_tx_head_ptr = uart_tx_buffer;
    if(--uart_tx_buf_cnt == 0)                     /* if buffer is empty */
      cbiBF(UCSRB,UDRIE);                                 /* disable UDRIE */
  }  
}

/* UART Receive IRQ */
SIGNAL(SIG_USART_RECV)
{
  //*uart_rx_tail_ptr=inp(UDR);                 /* read byte from recieve reg */
  *uart_rx_tail_ptr=UDR;                 /* read byte from recieve reg */
  uart_rx_buf_cnt++;
  if(++uart_rx_tail_ptr >= uart_rx_buffer+UART_BUF_SIZE) /*Wrap the pointer*/
    uart_rx_tail_ptr=uart_rx_buffer;
}

void UART_Putchar(unsigned char c)
{
    while(uart_tx_buf_cnt>=UART_BUF_SIZE);
    cli();
    uart_tx_buf_cnt++;
    *uart_tx_tail_ptr=c;                           /* store char in buffer */
    if(++uart_tx_tail_ptr >= uart_tx_buffer + UART_BUF_SIZE) /* Wrap ptr */
      uart_tx_tail_ptr=uart_tx_buffer;
    sbiBF(UCSRB,UDRIE);
    sei();
}

unsigned char UART_Getchar(void) {
  unsigned char c;
  c=0;  /* initialize c */
    while(uart_rx_buf_cnt==0);
  
    cli();
    uart_rx_buf_cnt--;
    c=*uart_rx_head_ptr;                         /* get char from buffer */
    if(++uart_rx_head_ptr >= uart_rx_buffer+UART_BUF_SIZE) /* Wrap ptr */
      uart_rx_head_ptr = uart_rx_buffer;

    sei();
  return c;
}

unsigned char UART_CharReady(void)
{
  return uart_rx_buf_cnt;
}
