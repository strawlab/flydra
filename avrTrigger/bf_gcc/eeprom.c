//***************************************************************************
//
//  File........: eeprom.c
//
//  Author(s)...: ATMEL Norway
//
//  Target(s)...: ATmega169
//
//  Compiler....: AVR-GCC 3.3.1; avr-libc 1.0
//
//  Description.: AVR Butterfly EEPROM routines
//
//  Revisions...: 1.0
//
//  YYYYMMDD - VER. - COMMENT                                       - SIGN.
//
//  20030116 - 1.0  - Created                                       - LHM
//  20031009          port to avr-gcc/avr-libc                      - M.Thomas
//
//***************************************************************************

#include <avr/io.h>
#include "eeprom.h"
// include workaround for ATmega169 and avr-libc 1.0
#include "eeprom169.h"

/*****************************************************************************
*
*   Function name : StoreEEPROM
*
*   Returns :       None
*
*   Parameters :    Pointer to string, number of bytes to write, adress in EEPROM
*
*   Purpose :       Write byte(s) to the EEPROM
*
*****************************************************************************/
void StoreEEPROM(char *pBuffer, char num_bytes, unsigned int EE_START_ADR)
{
    unsigned char i;
    for (i=0;i<num_bytes;i++) {
        // mt __EEPUT(EEPROM_START_ADR++, pBuffer[i]); // Store parameters
		eeprom_write_byte_169(&EE_START_ADR, pBuffer[i]);
		EE_START_ADR++;
	}
}


/*****************************************************************************
*
*   Function name : ReadEEPROM
*
*   Returns :       None
*
*   Parameters :    Pointer to string, number of bytes to read, adress in EEPROM
*
*   Purpose :       Write byte(s) to the EEPROM
*
*****************************************************************************/

void LoadEEPROM(char *pBuffer, char num_bytes, unsigned int EE_START_ADR)
{
    unsigned char i;
    for (i=0;i<num_bytes;i++) {
        // mt __EEGET(pBuffer[i], EEPROM_START_ADR++); // Load parameters
		pBuffer[i]=eeprom_read_byte_169(&EE_START_ADR);
		EE_START_ADR++;
	}
}
 

