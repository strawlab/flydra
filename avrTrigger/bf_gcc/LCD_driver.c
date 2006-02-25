//*****************************************************************************
//
//  File........: LCD_driver.c
//
//  Author(s)...: ATMEL Norway
//
//  Target(s)...: ATmega169
//
//  Compiler....: AVR-GCC 3.3.1; avr-libc 1.0
//
//  Description.: Functions used to control the AVR Butterfly LCD
//
//  Revisions...: 1.0
//
//  YYYYMMDD - VER. - COMMENT                                       - SIGN.
//
//  20021015 - 1.0  - Written for STK502                            - JLL
//  20030116 - 2.0  - Code adapted to AVR Butterfly                 - KS
//  20031009          port to avr-gcc/avr-libc                      - M.Thomas
//
//*****************************************************************************


// Include files.
#include <avr/io.h>
#include <avr/pgmspace.h>
#include <inttypes.h>
#include <avr/signal.h>
// mt - only for AUTO:
#include "main.h"
// mt - for gButtonTimeout
#include "button.h"
#include "LCD_driver.h"

#ifndef BOOL
#define BOOL    char
#define FALSE   0
#define TRUE    (!FALSE)
#endif

// Variable from "button.c" to prevent button-bouncing
extern unsigned char gButtonTimeout;    

volatile char gAutoPressJoystick = FALSE;

// Used to indicate when the LCD interrupt handler should update the LCD
// mt jw char gLCD_Update_Required = FALSE;
volatile char gLCD_Update_Required = FALSE;

// LCD display buffer (for double buffering).
volatile char LCD_Data[LCD_REGISTER_COUNT];

// Buffer that contains the text to be displayed
// Note: Bit 7 indicates that this character is flashing
volatile char gTextBuffer[TEXTBUFFER_SIZE];

// Only six letters can be shown on the LCD.
// With the gScroll and gScrollMode variables,
// one can select which part of the buffer to show
volatile signed char gScroll;
volatile char gScrollMode;

////Start-up delay before scrolling a string over the LCD
char gLCD_Start_Scroll_Timer = 0;

// The gFlashTimer is used to determine the on/off
// timing of flashing characters
volatile char gFlashTimer = 0;

// Turns on/off the colons on the LCD
char gColon = 0;


// Look-up table used when converting ASCII to
// LCD display data (segment control)
// mt __flash unsigned int LCD_character_table[] =
unsigned int LCD_character_table[] PROGMEM =
{
    0x0A51,     // '*' (?)
    0x2A80,     // '+'
    0x0000,     // ',' (Not defined)
    0x0A00,     // '-'
    0x0A51,     // '.' Degree sign
    0x0000,     // '/' (Not defined)
    0x5559,     // '0'
    0x0118,     // '1'
    0x1e11,     // '2
    0x1b11,     // '3
    0x0b50,     // '4
    0x1b41,     // '5
    0x1f41,     // '6
    0x0111,     // '7
    0x1f51,     // '8
    0x1b51,     // '9'
    0x0000,     // ':' (Not defined)
    0x0000,     // ';' (Not defined)
    0x0000,     // '<' (Not defined)
    0x0000,     // '=' (Not defined)
    0x0000,     // '>' (Not defined)
    0x0000,     // '?' (Not defined)
    0x0000,     // '@' (Not defined)
    0x0f51,     // 'A' (+ 'a')
    0x3991,     // 'B' (+ 'b')
    0x1441,     // 'C' (+ 'c')
    0x3191,     // 'D' (+ 'd')
    0x1e41,     // 'E' (+ 'e')
    0x0e41,     // 'F' (+ 'f')
    0x1d41,     // 'G' (+ 'g')
    0x0f50,     // 'H' (+ 'h')
    0x2080,     // 'I' (+ 'i')
    0x1510,     // 'J' (+ 'j')
    0x8648,     // 'K' (+ 'k')
    0x1440,     // 'L' (+ 'l')
    0x0578,     // 'M' (+ 'm')
    0x8570,     // 'N' (+ 'n')
    0x1551,     // 'O' (+ 'o')
    0x0e51,     // 'P' (+ 'p')
    0x9551,     // 'Q' (+ 'q')
    0x8e51,     // 'R' (+ 'r')
    0x9021,     // 'S' (+ 's')
    0x2081,     // 'T' (+ 't')
    0x1550,     // 'U' (+ 'u')
    0x4448,     // 'V' (+ 'v')
    0xc550,     // 'W' (+ 'w')
    0xc028,     // 'X' (+ 'x')
    0x2028,     // 'Y' (+ 'y')
    0x5009,     // 'Z' (+ 'z')
    0x0000,     // '[' (Not defined)
    0x0000,     // '\' (Not defined)
    0x0000,     // ']' (Not defined)
    0x0000,     // '^' (Not defined)
    0x0000      // '_'
};


/*****************************************************************************
*
*   Function name : LCD_Init
*
*   Returns :       None
*
*   Parameters :    None
*
*   Purpose :       Initialize LCD_displayData buffer.
*                   Set up the LCD (timing, contrast, etc.)
*
*****************************************************************************/
void LCD_Init (void)
{
    LCD_AllSegments(FALSE);                     // Clear segment buffer.

    LCD_CONTRAST_LEVEL(LCD_INITIAL_CONTRAST);    //Set the LCD contrast level

    // Select asynchronous clock source, enable all COM pins and enable all
    // segment pins.
    LCDCRB = (1<<LCDCS) | (3<<LCDMUX0) | (7<<LCDPM0);

    // Set LCD prescaler to give a framerate of 32,0 Hz
    LCDFRR = (0<<LCDPS0) | (7<<LCDCD0);    

    LCDCRA = (1<<LCDEN) | (1<<LCDAB);           // Enable LCD and set low power waveform

    //Enable LCD start of frame interrupt
    LCDCRA |= (1<<LCDIE);

    gLCD_Update_Required = FALSE;


}


/*****************************************************************************
*
*   Function name : LCD_WriteDigit(char c, char digit)
*
*   Returns :       None
*
*   Parameters :    Inputs
*                   c: The symbol to be displayed in a LCD digit
*                   digit: In which digit (0-5) the symbol should be displayed
*                   Note: Digit 0 is the first used digit on the LCD,
*                   i.e LCD digit 2
*
*   Purpose :       Stores LCD control data in the LCD_displayData buffer.
*                   (The LCD_displayData is latched in the LCD_SOF interrupt.)
*
*****************************************************************************/
void LCD_WriteDigit(char c, char digit)
{

    unsigned int seg = 0x0000;                  // Holds the segment pattern
    char mask, nibble;
    volatile char *ptr;
    char i;


    if (digit > 5)                              // Skip if digit is illegal
        return;

    //Lookup character table for segmet data
    if ((c >= '*') && (c <= 'z'))
    {
        // c is a letter
        if (c >= 'a')                           // Convert to upper case
            c &= ~0x20;                         // if necessarry

        c -= '*';

        //mt seg = LCD_character_table[c];
        seg = (unsigned int) pgm_read_word(&LCD_character_table[(uint8_t)c]); 
    }

    // Adjust mask according to LCD segment mapping
    if (digit & 0x01)
        mask = 0x0F;                // Digit 1, 3, 5
    else
        mask = 0xF0;                // Digit 0, 2, 4

    ptr = LCD_Data + (digit >> 1);  // digit = {0,0,1,1,2,2}

    for (i = 0; i < 4; i++)
    {
        nibble = seg & 0x000F;
        seg >>= 4;
        if (digit & 0x01)
            nibble <<= 4;
        *ptr = (*ptr & mask) | nibble;
        ptr += 5;
    }
}



/*****************************************************************************
*
*   Function name : LCD_AllSegments(unsigned char input)
*
*   Returns :       None
*
*   Parameters :    show -  [TRUE;FALSE]
*
*   Purpose :       shows or hide all all LCD segments on the LCD
*
*****************************************************************************/
void LCD_AllSegments(char show)
{
    unsigned char i;

    if (show)
        show = 0xFF;

    // Set/clear all bits in all LCD registers
    for (i=0; i < LCD_REGISTER_COUNT; i++)
        *(LCD_Data + i) = show;
}


/*****************************************************************************
*
*   LCD Interrupt Routine
*
*   Returns :       None
*
*   Parameters :    None
*
*   Purpose: Latch the LCD_displayData and Set LCD_status.updateComplete
*
*****************************************************************************/

SIGNAL(SIG_LCD)
{
    static char LCD_timer = LCD_TIMER_SEED;
    char c;
    char c_flash;
    char flash;

    char EOL;
    unsigned char i;

    static char timeout_count;
    static char auto_joystick_count;

    c_flash=0; // mt
    
/**************** Button timeout for the button.c, START ****************/
    if(!gButtonTimeout)
    {
        timeout_count++;
        
        if(timeout_count > 3)
        {
            gButtonTimeout = TRUE;
            timeout_count = 0;
        }
    }

/**************** Button timeout for the button.c, END ******************/

/**************** Auto press joystick for the main.c, START *************/

    if(gAutoPressJoystick == AUTO)
    {
        auto_joystick_count++;
        
        if(auto_joystick_count > 16)
        {
            gAutoPressJoystick = TRUE;
            auto_joystick_count = 15;
        }
    }
    else
        auto_joystick_count = 0;


/**************** Auto press joystick for the main.c, END ***************/    

    LCD_timer--;                    // Decreased every LCD frame

    if (gScrollMode)
    {
        // If we are in scroll mode, and the timer has expired,
        // we will update the LCD
        if (LCD_timer == 0)
        {
            if (gLCD_Start_Scroll_Timer == 0)
            {
                gLCD_Update_Required = TRUE;
            }
            else
                gLCD_Start_Scroll_Timer--;
        }
    }
    else    
    {   // if not scrolling,
        // disble LCD start of frame interrupt
//        cbi(LCDCRA, LCDIE);   //DEBUG
        gScroll = 0;
    }


    EOL = FALSE;
    if (gLCD_Update_Required == TRUE)
    {
        // Duty cycle of flashing characters
        if (gFlashTimer < (LCD_FLASH_SEED >> 1))
            flash = 0;
        else
            flash = 1;

        // Repeat for the six LCD characters
        for (i = 0; i < 6; i++)
        {
            if ((gScroll+i) >= 0 && (!EOL))
            {
                // We have some visible characters
                c = gTextBuffer[i + gScroll];
                c_flash = c & 0x80 ? 1 : 0;
                c = c & 0x7F;

                if (c == '\0')
                    EOL = i+1;      // End of character data
            }
            else
                c = ' ';

            // Check if this character is flashing

            if (c_flash && flash)
                LCD_WriteDigit(' ', i);
            else
                LCD_WriteDigit(c, i);
        }

        // Copy the segment buffer to the real segments
        for (i = 0; i < LCD_REGISTER_COUNT; i++)
            *(pLCDREG + i) = *(LCD_Data+i);

        // Handle colon
        if (gColon)
            *(pLCDREG + 8) = 0x01;
        else
            *(pLCDREG + 8) = 0x00;

        // If the text scrolled off the display,
        // we have to start over again.
        if (EOL == 1)
            gScroll = -6;
        else
            gScroll++;

        // No need to update anymore
        gLCD_Update_Required = FALSE;
    }


    // LCD_timer is used when scrolling text
    if (LCD_timer == 0)
    {
/*        if ((gScroll <= 0) || EOL)
            LCD_timer = LCD_TIMER_SEED/2;
        else*/
            LCD_timer = LCD_TIMER_SEED;
    }

    // gFlashTimer is used when flashing characters
    if (gFlashTimer == LCD_FLASH_SEED)
        gFlashTimer= 0;
    else
        gFlashTimer++;

}
