/*
	Local helper functions for the
	missing __flash workaround 
	
	- not part of the org. ATMEL code -
	done by Martin Thomas, KL, .de
*/

#ifndef _pgmspacehlp_h_
#define _pgmspacehlp_h_

#include <avr/pgmspace.h>

/* Read a float (4 Bytes) from PROGMEM
    the idea using a union is from Joerg Wunsch, found
    on the avr-gcc mailing-list, marked as "public domain"
    modifications by mt: (a) use pgm_read_word, (b) pass 
	pointer to float as param. and cast
*/
static inline float pgm_read_float_hlp(const float *addr)
{	
	union
	{
		int i[2];	// uint16_t 
		float f;
	} u;
	
	u.i[0]=pgm_read_word((PGM_P)addr);
	u.i[1]=pgm_read_word((PGM_P)addr+2);
	
	return u.f;
} 
#endif
