//*****************************************************************************
//
//  File........: LCD_functions.h
//
//  Author(s)...: ATMEL Norway
//
//  Target(s)...: ATmega169
//
//  Description.: Functions for LCD_functions.c
//
//  Revisions...: 1.0
//
//  YYYYMMDD - VER. - COMMENT                                       - SIGN.
//
//  20021015 - 1.0  - File created                                  - LHM
//  20031009          port to avr-gcc/avr-libc                      - M.Thomas
//
//*****************************************************************************

// mt
#include <avr/pgmspace.h>
//Functions
// mt void LCD_puts_f(char __flash *pFlashStr, char scrollmode);
// mt jw writes : ...(char *pFlahsStr...
void LCD_puts_f(const char *pFlashStr, char scrollmode);
void LCD_puts(char *pStr, char scrollmode);
void LCD_UpdateRequired(char update, char scrollmode);
//mt void LCD_putc(char digit, char character);
void LCD_putc(uint8_t digit, char character);
void LCD_Clear(void);
void LCD_Colon(char show);
void LCD_FlashReset(void);
char SetContrast(char input);
