#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/signal.h>
#include <stdio.h>

/* Buffersizes must be 2^n */

#define TBUFSIZE	32
#define RBUFSIZE	32

#define TMASK		(TBUFSIZE-1)
#define RMASK		(RBUFSIZE-1)

volatile unsigned char tbuf[TBUFSIZE];
volatile unsigned char rbuf[RBUFSIZE];

volatile unsigned char t_in;
volatile unsigned char t_out;

volatile unsigned char r_in;
volatile unsigned char r_out;

/* When modifing this code remember there is a difference between SIGNAL and
   INTERRUPT, the first instruction in the SIGNAL routine is 'cli' thus disabeling
   all interrupts while executing the ISR. INTERRUPT's first instruction is 'sei'
   allowing further interrupt processing */

SIGNAL(SIG_UART_RECV) {
	/* Receive interrupt */
	char c;
	
	c = UDR;							// Get received char

	rbuf[r_in & RMASK] = c;
	r_in++;
}

SIGNAL(SIG_UART_DATA) {
	/* Data register empty indicating next char can be transmitted */
	if(t_in != t_out) {
		UDR = tbuf[t_out & TMASK];
		t_out++;	
	}
	else {
		UCR &= ~(1<<UDRIE);
	}
}

char tbuflen(void) {
	return(t_in - t_out);
}

int UART_putchar(char c) {
	/* Fills the transmit buffer, if it is full wait */
	while((TBUFSIZE - tbuflen()) <= 2);
	
	/* Add data to the transmit buffer, enable TXCIE */
	tbuf[t_in & TMASK] = c;
	t_in++;
	
	UCR |= (1<<UDRIE);			// Enable UDR empty interrupt
	
	return(0);
}

char rbuflen(void) {
	return(r_in - r_out);
}

int UART_getchar(void) {
	unsigned char c;

	while(rbuflen() == 0);
	
	c = rbuf[r_out & RMASK];
	r_out++;
	
	return(c);
}

void UART_first_init(void) {
	/* First init for the UART */

	UBRR = 25;											// 19200 BPS
	UCR = (1<<RXCIE)|(1<<TXEN)|(1<<RXEN);			// 8 Databits, receive and transmit enabled, receive and transmit complete interrupt enabled
	
	fdevopen(UART_putchar, UART_getchar, 0);
	sei();												// Global interrupt enable
}




