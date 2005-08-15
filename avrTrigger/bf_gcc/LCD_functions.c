//*****************************************************************************
//
//  File........: LCD_functions.c
//
//  Author(s)...: ATMEL Norway
//
//  Target(s)...: ATmega169
//
//  Compiler....: AVR-GCC 3.3.1; avr-libc 1.0
//
//  Description.: Additional LCD functions, scrolling text and write data
//
//  Revisions...: 1.0
//
//  YYYYMMDD - VER. - COMMENT                                       - SIGN.
//
//  20021015 - 1.0  - Created                                       - LHM
//  20030116 - 2.0  - Code adapted to AVR Butterflyup               - KS
//  20031009          port to avr-gcc/avr-libc                      - M.Thomas
//
//*****************************************************************************

//  Include files
#include <avr/io.h>
#include <avr/pgmspace.h>
#include <inttypes.h>
#include "LCD_driver.h"
#include "LCD_functions.h"
#include "BCD.h"
// mt only for KEY_* and ST_OPTIONS_DISPLAY* definitions:
#include "main.h"


#define FALSE   0
#define TRUE    (!FALSE)

// mt char CONTRAST = LCD_INITIAL_CONTRAST;
uint8_t CONTRAST = LCD_INITIAL_CONTRAST;

// Start-up delay before scrolling a string over the LCD. "LCD_driver.c"
extern char gLCD_Start_Scroll_Timer;

/****************************************************************************
*
*	Function name : LCD_puts_f
*
*	Returns :		None
*
*	Parameters :	pFlashStr: Pointer to the string in flash
*                   scrollmode: Not in use
*
*	Purpose :		Writes a string stored in flash to the LCD
*
*****************************************************************************/

// mt void LCD_puts_f(char __flash *pFlashStr, char scrollmode)
void LCD_puts_f(const char *pFlashStr, char scrollmode)
{
    // char i;
	uint8_t i;

    while (gLCD_Update_Required);      // Wait for access to buffer

    // mt: for (i = 0; pFlashStr[i] && i < TEXTBUFFER_SIZE; i++)
	for (i = 0; (const char)(pgm_read_byte(&pFlashStr[i])) && i < TEXTBUFFER_SIZE; i++)
    {
        // mt: gTextBuffer[i] = pFlashStr[i];
		gTextBuffer[i] = pgm_read_byte(&pFlashStr[i]);
    }

    gTextBuffer[i] = '\0';

    if (i > 6)
    {
        gScrollMode = 1;        // Scroll if text is longer than display size
        gScroll = 0;
        gLCD_Start_Scroll_Timer = 3;    //Start-up delay before scrolling the text
    }
    else
    {
        gScrollMode = 0;        
        gScroll = 0;
    }

    gLCD_Update_Required = 1;
}


/****************************************************************************
*
*	Function name : LCD_puts
*
*	Returns :		None
*
*	Parameters :	pStr: Pointer to the string
*                   scrollmode: Not in use
*
*	Purpose :		Writes a string to the LCD
*
*****************************************************************************/
void LCD_puts(char *pStr, char scrollmode)
{
	uint8_t i; // char i;
	
	while (gLCD_Update_Required);      // Wait for access to buffer

    for (i = 0; pStr[i] && i < TEXTBUFFER_SIZE; i++)
    {
        gTextBuffer[i] = pStr[i];
    }

    gTextBuffer[i] = '\0';

    if (i > 6)
    {
        gScrollMode = 1;        // Scroll if text is longer than display size
        gScroll = 0;
        gLCD_Start_Scroll_Timer = 3;    //Start-up delay before scrolling the text
    }
    else
    {
        gScrollMode = 0;        
        gScroll = 0;
    }

    gLCD_Update_Required = 1;
}


/****************************************************************************
*
*	Function name : LCD_putc
*
*	Returns :		None
*
*	Parameters :	digit: Which digit to write on the LCD
*                   character: Character to write
*
*	Purpose :		Writes a character to the LCD
*
*****************************************************************************/
// mt void LCD_putc(char digit, char character)
void LCD_putc(uint8_t digit, char character)
{
    if (digit < TEXTBUFFER_SIZE)
        gTextBuffer[digit] = character;
}


/****************************************************************************
*
*	Function name : LCD_Clear
*
*	Returns :		None
*
*	Parameters :	None
*
*	Purpose :		Clear the LCD
*
*****************************************************************************/
void LCD_Clear(void)
{
    uint8_t i; // char i;
	   
    for (i=0; i<TEXTBUFFER_SIZE; i++)
        gTextBuffer[i] = ' ';
}


/****************************************************************************
*
*	Function name : LCD_Colon
*
*	Returns :		None
*
*	Parameters :	show: Enables the colon if TRUE, disable if FALSE
*
*	Purpose :		Enable/disable colons on the LCD
*
*****************************************************************************/
void LCD_Colon(char show)
{
    gColon = show;
}


/****************************************************************************
*
*	Function name : LCD_UpdateRequired
*
*	Returns :		None
*
*	Parameters :	update: TRUE/FALSE
*                   scrollmode: not in use
*
*	Purpose :		Tells the LCD that there is new data to be presented
*
*****************************************************************************/
void LCD_UpdateRequired(char update, char scrollmode)
{

    while (gLCD_Update_Required);
    
    gScrollMode = scrollmode;
    gScroll = 0;

    gLCD_Update_Required = update;
}


/****************************************************************************
*
*	Function name : LCD_FlashReset
*
*	Returns :		None
*
*	Parameters :	None
*
*	Purpose :		This function resets the blinking cycle of a flashing digit
*
*****************************************************************************/
void LCD_FlashReset(void)
{
    gFlashTimer = 0;
}



/****************************************************************************
*
*	Function name : SetContrast
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*	Purpose :		Adjust the LCD contrast
*
*****************************************************************************/
char SetContrast(char input)
{
    static char enter = 1;
    char CH, CL;

    if (enter)
    {
        LCD_Clear();
        enter = 0;
    }

    CH = CHAR2BCD2(CONTRAST);
    CL = (CH & 0x0F) + '0';
    CH = (CH >> 4) + '0';

    LCD_putc(0, 'C');
    LCD_putc(1, 'T');
    LCD_putc(2, 'R');
    LCD_putc(3, ' ');
    LCD_putc(4, CH);
    LCD_putc(5, CL);

    LCD_UpdateRequired(TRUE, 0);

    if (input == KEY_PLUS)
        CONTRAST++;
    else if (input == KEY_MINUS)
        CONTRAST--;

    if (CONTRAST == 255)
        CONTRAST = 0;
    if (CONTRAST > 15)
        CONTRAST = 15;

    LCD_CONTRAST_LEVEL(CONTRAST);


    if (input == KEY_ENTER)
    {
        enter = 1;
        return ST_OPTIONS_DISPLAY_CONTRAST;
    }

    return ST_OPTIONS_DISPLAY_CONTRAST_FUNC;
}


