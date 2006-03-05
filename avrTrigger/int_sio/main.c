#include <avr/io.h>
#include <avr/pgmspace.h>
#include <stdio.h>
#include "UART.h"

int main(void) {
    unsigned char s[20];

    UART_first_init();				// First init UART
	
	for(;;) {
		/* Same function as printf() but the format string resides in flash */
		printf_P(PSTR("Enter text\r\n"));
		
		while(scanf("%s",s) == 0);

		printf_P(PSTR("\r\nYou entered '%s'\r\n"),s);
	}
}

