//***************************************************************************
//
//  File........: vcard.c
//
//  Author(s)...: ATMEL Norway
//
//  Target(s)...: ATmega169
//
//  Compiler....: AVR-GCC 3.3.1; avr-libc 1.0
//
//  Description.: AVR Butterfly Name-tag
//
//  Revisions...: 1.0
//
//  YYYYMMDD - VER. - COMMENT                                       - SIGN.
//
//  20030116 - 1.0  - Created                                       - LHM
//  20031009          port to avr-gcc/avr-libc                      - M.Thomas
//  20031205          fixed store length to eeprom from RS232       - mt
//  20060106          fixed length-check in vCard()                 - mt
//
//***************************************************************************

//mtA
//#include <inavr.h>
//#include "iom169.h"
#include <avr/io.h>
#include <inttypes.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
//mtE
#include "main.h"
#include "button.h"
#include "LCD_functions.h"
#include "usart.h"
#include "eeprom.h"
#include "vcard.h"

// mt no eeprom-support for Mega169 in avr-libc (at least up to V1.2.3):
// no! // #include <avr/eeprom.h>
// include workaround:
#include "eeprom169.h"

// mt s/index/indexps
//char index = 0;         //variable to keep the lenght of the present string
uint8_t indexps = 0;

char gUART = FALSE;

char Name[STRLENGHT];

// mt __flash char TEXT_WAIT[]                     = "waiting for input on RS232";
// mt: jw method used 
/// const char TEXT_WAIT[] PROGMEM                = "waiting for input on RS232";

/*****************************************************************************
*
*   Function name : vCard
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Puts the name in EEPROM on the LCD
*
*****************************************************************************/
char vCard(char input)
{
    static char enter = 1;
    const uint16_t lenAddr=EEPROM_START;

    if (enter)
    {
        enter = 0;

        // mt __EEGET(indexps, EEPROM_START);           // Load the length if the name
        indexps=eeprom_read_byte_169(&lenAddr);
        
        if((indexps < 1) || (indexps > STRLENGHT))   // if illegal length / mt: fixed, was: |    
        {
            indexps = 0;
            Name[0] = 'A';
            
            enter = 1;
            return ST_VCARD_ENTER_NAME;               //enter new name
        }
        else
        {
            LoadEEPROM(Name, indexps, EEPROM_START + 1);  // Load name 
            LCD_puts(Name, 1);
        }
    }

    else if (input == KEY_NEXT)
    {
        enter = 1;
        return ST_VCARD_ENTER_NAME;
    }
    else if (input == KEY_PREV)
    {
        enter = 1;
        return ST_VCARD;
    }
    
    return ST_VCARD_FUNC;
}


/*****************************************************************************
*
*   Function name : EnterName
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Lets the user enter a name using the joystick. Pressing the
*                   joystick UP/DOWN will browse the alphabet and NEXT/PREV 
*                   will shift between the characters in the name.
*
*****************************************************************************/
char EnterName(char input)
{
    static char enter = 1;

    // mt static char temp_index = 0;
    static uint8_t temp_index = 0;
    static char temp_name[6];
    
    // mt char i;
    static uint8_t i;
    uint16_t lenAddr=EEPROM_START;

    if (enter)
    {
        LoadEEPROM(Name, indexps, EEPROM_START + 1);  // Load name from EEPROM
                
        if(indexps)
            indexps -= 1;         //make the last character in name blink

        enter = 0;        
    }
    else
    {
        temp_index = indexps;
                
        for(i = 5; (i != 255); i--, temp_index--)
        {
            if ((Name[temp_index] >= ' ') && (Name[temp_index] <= 'z') && (temp_index != 255)) //check if it's legal character
                temp_name[i] = Name[temp_index];
            else
                temp_name[i] = ' '; // if not, put in a space
        }       
        
        LCD_putc(0, temp_name[0]);
        LCD_putc(1, temp_name[1]);
        LCD_putc(2, temp_name[2]);
        LCD_putc(3, temp_name[3]);
        LCD_putc(4, temp_name[4]);
        LCD_putc(5, temp_name[5] | 0x80);   //Make this digit blink
        LCD_putc(6, '\0');

        if (input != KEY_NULL)
            LCD_FlashReset();
     
        LCD_UpdateRequired(TRUE, 0);
    }
    
    if (input != KEY_NULL)
        LCD_FlashReset();

    if ( input == KEY_MINUS ) // mt 1/06 (input == KEY_PLUS)
    {
       
        Name[indexps]--;

        if( (('!' <= Name[indexps]) && (Name[indexps] <= '/')) && (Name[indexps] != ' '))
            Name[indexps] = ' ';
        else if( (':' <= Name[indexps]) && (Name[indexps] <= '@'))
            Name[indexps] = '9';
        else if(Name[indexps] >= '[')
            Name[indexps] = 'Z';
        else if(Name[indexps] < ' ')
            Name[indexps] = 'Z';

    }
    else if ( input == KEY_PLUS ) // mt 1/06 (input == KEY_MINUS)
    {
        Name[indexps]++;

        if( (('!' <= Name[indexps]) && (Name[indexps] <= '/')) && (Name[indexps] != ' '))
            Name[indexps] = '0';
        else if( (':' <= Name[indexps]) && (Name[indexps] <= '@'))
            Name[indexps] = 'A';
        else if(Name[indexps] >= '[')
            Name[indexps] = ' ';
        else if(Name[indexps] < ' ')
            Name[indexps] = ' ';
    }
    else if (input == KEY_PREV)
    {
        if(indexps)
        {
            indexps--;
        }
    }
    else if (input == KEY_NEXT)
    {
        if(indexps < STRLENGHT)
        {
			i = Name[indexps]; // ext. mt 1/2006
            indexps++;
            Name[indexps] = i; // Name[indexps] = 'A';
        }
    }
    else if (input == KEY_ENTER)
    {
        indexps++;
        
        Name[indexps] = '\0';
        
        // mt __EEPUT(EEPROM_START, indexps);   //store the length of name in EEPROM
        eeprom_write_byte_169(&lenAddr,indexps);
    
        StoreEEPROM(Name, indexps, EEPROM_START + 1);  //store the Name in EEPROM
        
        enter = 1;
        return ST_VCARD_FUNC;
    }

    return ST_VCARD_ENTER_NAME_FUNC;
}



/*****************************************************************************
*
*   Function name : RS232
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Store data from the UART to EEPROM
*
*****************************************************************************/
char RS232(char input)
{
    static char enter = 1;
    char c;
    static char buffer[STRLENGHT];
    //static char temp_index;
    static uint8_t temp_index;
    uint16_t lenAddr=EEPROM_START;
    
    if (enter)
    {
        cli(); // mt __disable_interrupt();
        
        // boost IntRC to 2Mhz to achieve 19200 baudrate
        CLKPR = (1<<CLKPCE);        // set Clock Prescaler Change Enable
        // set prescaler = 4, Inter RC 8Mhz / 4 = 2Mhz
        CLKPR = (1<<CLKPS1);
        
        sei(); // mt __enable_interrupt();
        
        // mt jw-meth: LCD_puts_f(TEXT_WAIT, 0);
        LCD_puts_f(PSTR("waiting for input on RS232"), 0);
        enter = 0;
        temp_index = 0;
        c = UDR;                       // Dummy read to clear receive buffer
        gUART = TRUE;
    }

    if (UCSRA & (1<<RXC))
    {
        c = UDR;
        if (c != '\r')
        {
            if (temp_index < STRLENGHT)
                buffer[temp_index++] = c;
        }
        else    // UART transmission completed
        {
            cli(); // mt __disable_interrupt();
                
            CLKPR = (1<<CLKPCE);        // set Clock Prescaler Change Enable
            // set prescaler = 8, Inter RC 8Mhz / 8 = 1Mhz
            CLKPR = (1<<CLKPS1) | (1<<CLKPS0);
                
            sei(); // mt __enable_interrupt();
                
            if(temp_index)   
            {
                buffer[temp_index] = '\0';
                for (temp_index = 0; buffer[temp_index]; temp_index++)
                    Name[temp_index] = buffer[temp_index];
                Name[temp_index] = '\0';
                
                enter = 1;
                        
                // mt __EEPUT(EEPROM_START, temp_index);   //store the length of name in EEPROM
                eeprom_write_byte_169(&lenAddr,temp_index);
                StoreEEPROM(Name, temp_index, EEPROM_START + 1);  //store the Name in EEPROM
                
                indexps = temp_index;
                
                gUART = FALSE;               
                return ST_VCARD_FUNC;
            }
            else    // if no characters received 
            {
                enter = 1;            
                return ST_VCARD_DOWNLOAD_NAME;
            }
        }
    }


    if (input != KEY_NULL)
    {
        enter = 1;
        
        cli(); // mt __disable_interrupt();
        
        CLKPR = (1<<CLKPCE);        // set Clock Prescaler Change Enable
        // set prescaler = 8, Inter RC 8Mhz / 8 = 1Mhz
        CLKPR = (1<<CLKPS1) | (1<<CLKPS0);
        
        sei(); // __enable_interrupt();
        
        gUART = FALSE;
        return ST_VCARD_DOWNLOAD_NAME;
    }


    return ST_VCARD_DOWNLOAD_NAME_FUNC;
}
