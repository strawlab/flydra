//***************************************************************************
//
//  File........: test.c
//
//  Author(s)...: ATMEL Norway
//
//  Target(s)...: ATmega169
//
//  Compiler....: AVR-GCC 3.3.1; avr-libc 1.0
//
//  Description.: AVR Butterfly TEST routines
//
//  Revisions...: 1.0
//
//  YYYYMMDD - VER. - COMMENT                                       - SIGN.
//
//  20030116 - 1.0  - Created                                       - LHM
//  20031009          port to avr-gcc/avr-libc                      - M.Thomas
//  20060107          TestWaitEnter()                               - mt
//
//***************************************************************************

//mtA
//#include <inavr.h>
//#include "iom169.h"
#include <avr/io.h>
#include <avr/wdt.h>
//mtE
#include "main.h"
#include "test.h"
#include "button.h"
#include "LCD_driver.h"
#include "LCD_functions.h"
#include "sound.h"
#include "ADC.h"
#include "RTC.h"
#include "dataflash.h"

#define pLCDREG_test (*(char *)(0xEC))

// mtA
static void TestWaitEnter(void)
{
	char input = 0;
    while(input != KEY_ENTER)
    {
        input = getkey();           // Read buttons
    }
}
// mtE

void Test(void)
{
    char input = 0;
    char i;
    char j;
    
    int integer;
    int temp_hi;
    
    TIMSK2 = 0; // Stop clock, the interrupt will disturb the sound-test cause
                // the play-routine doesn't use interrupt...

//mtA    
    LCD_puts_f(PSTR("Test"),0); // mt LCD_puts("Test",0);

#if 0        
    while(input != KEY_ENTER)
    {
        input = getkey();           // Read buttons
    }
#endif
	TestWaitEnter();
//mtE


//// //// //// //// JOYSTICK TEST //// //// //// ////

    LCD_puts_f(PSTR("Left"),0);// mt LCD_puts("Left",0);
    
    while(input != KEY_PREV)
    {
        input = getkey();           // Read buttons
    }
    
    LCD_puts_f(PSTR("Up"),0);// mt LCD_puts("Up",0);
    
    while(input != KEY_PLUS)
    {
        input = getkey();           // Read buttons
    }
    
    LCD_puts_f(PSTR("Right"),0);// mt LCD_puts("Right",0);
    
    while(input != KEY_NEXT)
    {
        input = getkey();           // Read buttons
    }
    
    LCD_puts_f(PSTR("Down"),0);// mt LCD_puts("Down",0);
    
    while(input != KEY_MINUS)
    {
        input = getkey();           // Read buttons
    }



//// //// //// //// SOUND TEST //// //// //// ////
    
    LCD_puts_f(PSTR("Sound"),0);// mt LCD_puts("Sound",0);

//mtA
#if 0    
    input = 0;
    
    while(input != KEY_ENTER)
    {
        input = getkey();           // Read buttons
    }
#endif
	TestWaitEnter();
//mtE
        
    
    TCCR1A = (1<<COM1A1);// | (1<<COM1A0); // Set OC1A when upcounting, clear when downcounting
    TCCR1B = (1<<WGM13) | (1<<CS10);        // Phase/Freq-correct PWM, top value = ICR1
    
    sbiBF(TCCR1B, 0);             // start Timer1, prescaler(1)    
    
    OCR1AH = 0;     // Set a initial value in the OCR1A-register
    OCR1AL = 80;     // This will adjust the volume on the buzzer, lower value => higher volume
	
    for(integer = 200; integer >= 100; integer -= 5)
    {
        temp_hi = integer;
        
        ICR1H = (temp_hi >>= 8);
        ICR1L = integer;
    
        Delay(30);
    } 


    TCCR1A = 0;
    TCCR1B = 0;     //Stop timer1
 

//// //// //// //// LCD TEST //// //// //// ////    
    
    LCD_puts_f(PSTR("LCD test"),0);// mt LCD_puts("LCD test",0);
    
    // mt while(!(PINB & (1<<PORTB4)));
	while(!(PINB & (1<<PINB4)));

//mtA
#if 0
    input = 0;
    
    while(input != KEY_ENTER)
    {
        input = getkey();           // Read buttons
    }
#endif
	TestWaitEnter();
//mtE
    
    cbiBF(LCDCRA, LCDIE);
    
    for(i = 0; i < 20; i++) // set all LCD segment register to the variable ucSegments
    {
        *(&pLCDREG_test + i) = 0xFF;
    }
    
    // mt while(!(PINB & (1<<PORTB4)));   
	while(!(PINB & (1<<PINB4)));   
    
    Delay(100); 
        
    // mt while((PINB & (1<<PORTB4)));    //pull PINB cause the LCD interrupt is disabled which 
	while((PINB & (1<<PINB4)));    //pull PINB cause the LCD interrupt is disabled which 
                                    //controls the button-bouncing
    sbiBF(LCDCRA, LCDIE);




//// //// //// //// TEMPERATURE TEST //// //// //// ////
      
    ADC_init(0);

    integer = ADC_read();
    
    //check if temperature is between 21 and 30 degree Celcius
    if( (integer < 452) || (integer > 586) )  
    {
	
		LCD_puts_f(PSTR("Error Temperature"),0);// mt LCD_puts("Error Temperature",0);
		
        ErrorBeep();
    
//mtA
#if 0
	    input = 0;
    
	    while(input != KEY_ENTER)
	    {
	        input = getkey();           // Read buttons
	    }
#endif
		TestWaitEnter();
//mtE

        ADC_periphery();
  
        LCD_UpdateRequired(TRUE, 0);   
    
//mtA
#if 0
	    input = 0;
    
	    while(input != KEY_ENTER)
	    {
	        input = getkey();           // Read buttons
	    }
#endif
		TestWaitEnter();
//mtE
    }
    
 
//// //// //// //// Voltage TEST //// //// //// ////    

    ADC_init(1);
    
    integer = ADC_read();
    
    //if measuring outside 4,8 to 5,2 V
    if( (integer < 298) || (integer > 320) )
    {

        LCD_puts_f(PSTR("Error Voltage"),0);// mt LCD_puts("Error Voltage",0);

        ErrorBeep();

//mtA
#if 0
	    input = 0;
    
	    while(input != KEY_ENTER)
	    {
	        input = getkey();           // Read buttons
	    }
#endif
		TestWaitEnter();
//mtE
                
        ADC_periphery();
  
        LCD_UpdateRequired(TRUE, 0);   
    
//mtA
#if 0
	    input = 0;
    
	    while(input != KEY_ENTER)
	    {
	        input = getkey();           // Read buttons
	    }
#endif
		TestWaitEnter();
//mtE
    }

    
//// //// //// //// LIGHT TEST //// //// //// ////    

    LCD_puts_f(PSTR("Light"),0);// mt LCD_puts("Light",0);

//mtA
#if 0
    input = 0;

    while(input != KEY_ENTER)
    {
        input = getkey();           // Read buttons
    }
#endif
	TestWaitEnter();
//mtE
    
    ADC_init(2);

    input = 0;
    
    while(input != KEY_ENTER)
    {
        ADC_read();
        
        ADC_periphery();
    
        // mt __no_operation();
		asm volatile ("nop"::);
    
        LCD_UpdateRequired(TRUE, 0);    
        
        input = getkey();           // Read buttons
    
    }

    
    DF_CS_inactive;
    DF_CS_active;  
    
    for(j=0;j<50;j++)
    Buffer_Write_Byte (1,j,j);
        
    Buffer_To_Page (1,1);             //transfer to DataFlash


    DF_CS_inactive;
    DF_CS_active;    
    
    Cont_Flash_Read_Enable (1,0);

    for(j=0;j<50;j++)
    {
        
        if(j != DF_SPI_RW(0x00) )
        {
            LCD_puts_f(PSTR("Error DataFlash"),0);// mt LCD_puts("Error Dataflash",0);

            ErrorBeep();

//mtA
#if 0
		    input = 0;

    		while(input != KEY_ENTER)
    		{
        		input = getkey();           // Read buttons
    		}
#endif
			TestWaitEnter();
//mtE
            
            break;
        }
    }

    DF_CS_inactive;  



//// //// //// //// Display version# //// //// //// ////  

    i = Revision(1); // dummy readout

//mtA
#if 0
    input = 0;

    while(input != KEY_ENTER)
    {
        input = getkey();           // Read buttons
    }
#endif
	TestWaitEnter();
//mtE


//// //// //// //// RESET //// //// //// ////

    
	WDTCR = (1<<WDCE) | (1<<WDE);   //Enable Watchdog Timer to give reset
	// mtA 
	/// avr-libc meth. : "wdt_enable(WDTO_30MS);" but throws compiler error TODO
	// mtE
    
    while(1);                       //wait for reset...
}

void ErrorBeep(void)
{
    TCCR1A = (1<<COM1A1);// | (1<<COM1A0); // Set OC1A when upcounting, clear when downcounting
    TCCR1B = (1<<WGM13) | (1<<CS10);        // Phase/Freq-correct PWM, top value = ICR1
    
    sbiBF(TCCR1B, 0);             // start Timer1, prescaler(1)    
    
	OCR1AH = 0;     // Set a initial value in the OCR1A-register
    OCR1AL = 80;     // This will adjust the volume on the buzzer, lower value => higher volume
    
    ICR1H = 1;
    ICR1L = 0x3F;
        
    Delay(50);
           
    ICR1H = 1;
    ICR1L = 0xDE;
        
    Delay(300);
    
    TCCR1A = 0;
    TCCR1B = 0;    
}
