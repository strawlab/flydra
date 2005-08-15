//***************************************************************************
//
//  File........: main.c
//
//  Author(s)...: ATMEL Norway
//
//  Target(s)...: ATmega169
//
//  Compiler....: AVR-GCC 3.3.1; avr-libc 1.0
//
//  Description.: AVR Butterfly main module
//
//  Revisions...: 1.0
//
//  YYYYMMDD - VER. - COMMENT                                       - SIGN.
//
//  20030116 - 1.0  - Created                                       - KS
//  20031009          port to avr-gcc/avr-libc                      - M.Thomas (*)
//  20031204          fixed imcompatibility with sleep-modes        - mt
//  20040218          fixed 'logical and' in calibration            - shc/mt
//  20050827          fixed avr-libc iom169.h compatibility
//                    added keyklick function (from version6)       - mt/v6
//
//***************************************************************************

// (*) Martin Thomas, Kaiserslautern, Germany, e-mail: mthomas(at)rhrk.uni-kl.de 
// or eversmith(at)heizung-thomas.de
//
// I'm not working for ATMEL.
// The port is based on REV_06 of the ATMEL-Code (for IAR-C)
// Initialy I marked my changes with "// mt" or enclosed them with "// mtA" and 
// "// mtE" but forgot this for some changes esp. during debugging. 'diff' 
// against the original code to see everything that has been changed.

//mtA
//#include <inavr.h>
//#include "iom169.h"
#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>
#include <avr/sleep.h>
#include <inttypes.h>
//mtE

#include "main.h"
#include "LCD_functions.h"
#include "LCD_driver.h"
#include "button.h"
#include "RTC.h"
#include "timer0.h"
#include "BCD.h"
#include "usart.h"
#include "sound.h"
#include "ADC.h"
#include "dataflash.h"
//#include "eeprom.h"
// mt Test() is not realy needed 
//- can not be accessed without external hardware
#include "test.h"	
#include "vcard.h"
#include "menu.h"

#define pLCDREG_test (*(char *)(0xEC))

//mt: extern __flash unsigned int LCD_character_table[];
//    but this is not used here anyway...
extern unsigned int LCD_character_table[] PROGMEM;

extern volatile uint8_t gPowerSaveTimer;    // external Counter from "RTC.c"
char PowerSaveTimeout = 30;     // Initial value, enable power save mode after 30 min
BOOL AutoPowerSave = TRUE;      // Variable to enable/disable the Auto Power Save func
BOOL KeyClickStatus = FALSE;    // Variable to enable/disable keyclick

char gAutoPressJoystick = FALSE;    // global variable used in "LCD_driver.c"

char PowerSave = FALSE;         // 

char gPlaying = FALSE;           // global variable from "sound.c". To prevent  
                                // entering power save, when playing.

char gUART = FALSE;      // global variable from "vcard.c". To prevent 
                                // entering power save, when using the UART

unsigned char state;            // helds the current state, according to 
                                // "menu.h"



/*****************************************************************************
*
*   Function name : main
*
*   Returns :       None
*
*   Parameters :    None
*
*   Purpose :       Contains the main loop of the program
*
*****************************************************************************/
// mt __C_task void main(void)
int main(void)
{    
//    unsigned char state, nextstate;
    unsigned char nextstate;
    // mt static char __flash *statetext;
	PGM_P statetext;
	char (*pStateFunc)(char);
    char input;
    uint8_t i; // char i;
    char buttons;
    char last_buttons;
	
	last_buttons='\0';	// mt

    // Initial state variables
    state = ST_AVRBF;
    nextstate = ST_AVRBF;
    statetext = MT_AVRBF;
    pStateFunc = NULL;


    // Program initalization
    Initialization();
    sei(); // mt __enable_interrupt();
		
    for (;;)            // Main loop
    {
        if(!PowerSave)          // Do not enter main loop in power-save
        {
            // Plain menu text
            if (statetext)
            {
                LCD_puts_f(statetext, 1);
                LCD_Colon(0);
                statetext = NULL;
            }
    
    
            input = getkey();           // Read buttons
    
    
            if (pStateFunc)
            {
                // When in this state, we must call the state function
                nextstate = pStateFunc(input);
            }
            else if (input != KEY_NULL)
            {
                // Plain menu, clock the state machine
                nextstate = StateMachine(state, input);
            }
    
            if (nextstate != state)
            {
                state = nextstate;
                // mt: for (i=0; menu_state[i].state; i++)
				for (i=0; pgm_read_byte(&menu_state[i].state); i++)
                {
                    //mt: if (menu_state[i].state == state)
					if (pgm_read_byte(&menu_state[i].state) == state)
                    {
						// mtA
                        // mt - original: statetext =  menu_state[i].pText;
                        // mt - original: pStateFunc = menu_state[i].pFunc;
						/// mt this is like the example from an avr-gcc guru (mailing-list):
						statetext =  (PGM_P) pgm_read_word(&menu_state[i].pText);
						// mt - store pointer to function from menu_state[i].pFunc in pStateFunc
                        //// pStateFunc = pmttemp;	// oh je - wie soll ich das jemals debuggen - ?
						pStateFunc = (PGM_VOID_P) pgm_read_word(&menu_state[i].pFunc);
						// mtE
                        break;
                    }
                }
            }
        }
        
        
        //enable ATmega169 power save modus if autopowersave
        if(AutoPowerSave)
        {
            if(gPowerSaveTimer >= PowerSaveTimeout)
            {
                state = ST_AVRBF;
                gPowerSaveTimer = 0;
                PowerSave = TRUE;
            }
        }
        
        
        // If the joystick is held in the UP and DOWN position at the same time,
        // activate test-mode
		// mtA
        // if( !(PINB & (1<<PORTB7)) && !(PINB & (1<<PORTB6)) )    
		if( !(PINB & (1<<PINB7)) && !(PINB & (1<<PINB6)) )    
            Test();    
		// mtE
        
        // Check if the joystick has been in the same position for some time, 
        // then activate auto press of the joystick
        buttons = (~PINB) & PINB_MASK;
        buttons |= (~PINE) & PINE_MASK;
        
        if( buttons != last_buttons ) 
        {
            last_buttons = buttons;
            gAutoPressJoystick = FALSE;
        }
        else if( buttons )
        {
            if( gAutoPressJoystick == TRUE)
            {
                PinChangeInterrupt();
                gAutoPressJoystick = AUTO;
            }
            else    
                gAutoPressJoystick = AUTO;
        }

        
        
        // go to SLEEP
        if(!gPlaying && !gUART)              // Do not enter Power save if using UART or playing tunes
        {
            if(PowerSave)
                cbiBF(LCDCRA, 7);             // disable LCD
        
			// mtA
            SMCR = (3<<SM0) | (1<<SE);      // Enable Power-save mode
			asm volatile ("sleep"::);
            // __sleep();                      // Go to sleep
			// mt 20031204 - avr-libc 1.0 sleep.h seems to be incompatible with mega169 
			/// no! // set_sleep_mode(SLEEP_MODE_PWR_SAVE);
			/// no! // sleep_mode();
			// mtE
            
            if(PowerSave)
            {
                if(!(PINB & 0x40))              // press UP to wake from SLEEP
                {
                    PowerSave = FALSE;
                    
                    for(i = 0; i < 20; i++) // set all LCD segment register to the variable ucSegments
                    {
                        *(&pLCDREG_test + i) = 0x00;
                    }
                    
                    sbiBF(LCDCRA, 7);             // enable LCD
                    input = getkey();           // Read buttons
                }
            }
        }
        else
        {		
				// mtA
                SMCR = (1<<SE);                 // Enable idle mode
				asm volatile ("sleep"::);
                //__sleep();                      // Go to sleep        
				// mt 20031204 - avr-libc 1.0 sleep.h seems to be incompatible with mega169 
				/// no! // set_sleep_mode(SLEEP_MODE_IDLE);
				/// no! // sleep_mode();
				// mtE
            
        }   

        SMCR = 0;                       // Just woke, disable sleep

    } //End Main loop
	
	return 0; // mt 
}




/*****************************************************************************
*
*   Function name : StateMachine
*
*   Returns :       nextstate
*
*   Parameters :    state, stimuli
*
*   Purpose :       Shifts between the different states
*
*****************************************************************************/
unsigned char StateMachine(char state, unsigned char stimuli)
{
    unsigned char nextstate = state;    // Default stay in same state
    unsigned char i;

    // mt: for (i=0; menu_nextstate[i].state; i++)
	for (i=0; pgm_read_byte(&menu_nextstate[i].state); i++)
    {
        // mt: if (menu_nextstate[i].state == state && menu_nextstate[i].input == stimuli)
		if (pgm_read_byte(&menu_nextstate[i].state) == state && 
			pgm_read_byte(&menu_nextstate[i].input) == stimuli)
        {
            // This is the one!
            // mt: nextstate = menu_nextstate[i].nextstate;
			nextstate = pgm_read_byte(&menu_nextstate[i].nextstate);
            break;
        }
    }

    return nextstate;
}




/*****************************************************************************
*
*   Function name : Initialization
*
*   Returns :       None
*
*   Parameters :    None
*
*   Purpose :       Initializate the different modules
*
*****************************************************************************/
void Initialization(void)
{
    char tst;           // dummy

    OSCCAL_calibration();       // calibrate the OSCCAL byte
        
    CLKPR = (1<<CLKPCE);        // set Clock Prescaler Change Enable

    // set prescaler = 8, Inter RC 8Mhz / 8 = 1Mhz
    CLKPR = (1<<CLKPS1) | (1<<CLKPS0);

    // Disable Analog Comparator (power save)
    ACSR = (1<<ACD);

    // Disable Digital input on PF0-2 (power save)
    DIDR1 = (7<<ADC0D);

    // mt PORTB = (15<<PORTB0);       // Enable pullup on 
	PORTB = (15<<PB0);       // Enable pullup on 
    // mt PORTE = (15<<PORTE4);
	PORTE = (15<<PE4);

    sbiBF(DDRB, 5);               // set OC1A as output
    sbiBF(PORTB, 5);              // set OC1A high
            
    Button_Init();              // Initialize pin change interrupt on joystick
    
    RTC_init();                 // Start timer2 asynchronous, used for RTC clock

    Timer0_Init();              // Used when playing music etc.

    USART_Init(12);             // Baud rate = 9600bps
    
    DF_SPI_init();              // init the SPI interface to communicate with the DataFlash
    
    tst = Read_DF_status();

    DF_CS_inactive;             // disable DataFlash
        
    LCD_Init();                 // initialize the LCD
}





/*****************************************************************************
*
*   Function name : BootFunc
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Reset the ATmega169 which will cause it to start up in the 
*                   Bootloader-section. (the BOOTRST-fuse must be programmed)
*
*****************************************************************************/
// mt __flash char TEXT_BOOT[]                     
// mt - as in jw-patch: const char TEXT_BOOT[] PROGMEM	= "Jump to bootloader";

char BootFunc(char input)
{
    static char enter = 1;
    
    if(enter)
    {
        enter = 0;
        // mt jw LCD_puts_f(TEXT_BOOT, 1);
		LCD_puts_f(PSTR("Jump to bootloader"), 1);
    }
    else if(input == KEY_ENTER)
    {
        WDTCR = (1<<WDCE) | (1<<WDE);     //Enable Watchdog Timer to give reset
        while(1);   // wait for watchdog-reset, since the BOOTRST-fuse is 
                    // programmed, the Boot-section will be entered upon reset.
    }
    else if (input == KEY_PREV)
    {
        enter = 1;
        return ST_OPTIONS_BOOT;
    }
    
    return ST_OPTIONS_BOOT_FUNC;
}





/*****************************************************************************
*
*   Function name : PowerSaveFunc
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Enable power save
*
*****************************************************************************/
// mt __flash char TEXT_POWER[]                     = "Press enter to sleep";
// mt jw const char TEXT_POWER[]  PROGMEM  = "Press enter to sleep";

char PowerSaveFunc(char input)
{
    static char enter = 1;    
    
    if(enter)
    {
        enter = 0;
        //mt jw LCD_puts_f(TEXT_POWER, 1);
		LCD_puts_f(PSTR("Press enter to sleep"), 1);
    }
    else if(input == KEY_ENTER)
    {
        PowerSave = TRUE;
        enter = 1;
        return ST_AVRBF;
    }
    else if (input == KEY_PREV)
    {
        enter = 1;
        return ST_OPTIONS_POWER_SAVE;
    }
        
    return ST_OPTIONS_POWER_SAVE_FUNC;

}




/*****************************************************************************
*
*   Function name : AutoPower
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Enable/Disable auto power save
*
*****************************************************************************/
char AutoPower(char input)
{
    static char enter = 1;    
    
    char PH;
    char PL;
    
    if(enter)
    {
        enter = 0;
        
        if(AutoPowerSave)  
        {     
            PH = CHAR2BCD2(PowerSaveTimeout);
            PL = (PH & 0x0F) + '0';
            PH = (PH >> 4) + '0';
                
            LCD_putc(0, 'M');
            LCD_putc(1, 'I');
            LCD_putc(2, 'N');
            LCD_putc(3, ' ');
            LCD_putc(4, PH);
            LCD_putc(5, PL);
            LCD_putc(6, '\0');
        
            LCD_UpdateRequired(TRUE, 0);    
        }
        else
            LCD_puts_f(PSTR("Off"),1);	// mt LCD_puts("Off", 1);        
                    
    }
    else if(input == KEY_ENTER)
    {
         enter = 1;

         return ST_OPTIONS_AUTO_POWER_SAVE;
    }
    else if (input == KEY_PLUS)
    {

        PowerSaveTimeout += 5;
         
        if(PowerSaveTimeout > 90)
        {
            PowerSaveTimeout = 90;
        }
        else
        {    
            AutoPowerSave = TRUE;
           
            PH = CHAR2BCD2(PowerSaveTimeout);
            PL = (PH & 0x0F) + '0';
            PH = (PH >> 4) + '0';
                
            LCD_putc(0, 'M');
            LCD_putc(1, 'I');
            LCD_putc(2, 'N');
            LCD_putc(3, ' ');
            LCD_putc(4, PH);
            LCD_putc(5, PL);
            LCD_putc(6, '\0');
        
            LCD_UpdateRequired(TRUE, 0);        
        }
    }
    else if (input == KEY_MINUS)
    {
        if(PowerSaveTimeout)
            PowerSaveTimeout -= 5;

        if(PowerSaveTimeout < 5)
        {
            AutoPowerSave = FALSE;
            PowerSaveTimeout = 0;
            LCD_puts_f(PSTR("Off"),1);	// mt LCD_puts("Off", 1);
        }
        else
        {   
            AutoPowerSave = TRUE;
                      
            PH = CHAR2BCD2(PowerSaveTimeout);
            PL = (PH & 0x0F) + '0';
            PH = (PH >> 4) + '0';
            
            LCD_putc(0, 'M');
            LCD_putc(1, 'I');
            LCD_putc(2, 'N');
            LCD_putc(3, ' ');
            LCD_putc(4, PH);
            LCD_putc(5, PL);
            LCD_putc(6, '\0');
        
            LCD_UpdateRequired(TRUE, 0);                     
        }
    }
        
    return ST_OPTIONS_AUTO_POWER_SAVE_FUNC;    

}




/*****************************************************************************
*
*   Function name : KeyClick
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Enable/Disable keyclick
*
*****************************************************************************/
char KeyClick(char input)
{
    if(input == KEY_ENTER)
    {
         return ST_OPTIONS_KEYCLICK;
    }

    if ((input == KEY_PLUS) || (input == KEY_MINUS))
        KeyClickStatus = ~KeyClickStatus;

    if(KeyClickStatus)  
        LCD_puts_f(PSTR("On"),1);
    else
        LCD_puts_f(PSTR("Off"),1);
      
    LCD_UpdateRequired(TRUE, 0);                     
        
    return ST_OPTIONS_KEYCLICK_FUNC;    

}




/*****************************************************************************
*
*   Function name : Delay
*
*   Returns :       None
*
*   Parameters :    unsigned int millisec
*
*   Purpose :       Delay-loop
*
*****************************************************************************/
void Delay(unsigned int millisec)
{
    // mt, int i did not work in the simulator:  int i; 
	uint8_t i;
    
    while (millisec--)
        	//mt: for (i=0; i<125; i++);
			for (i=0; i<125; i++)  
				asm volatile ("nop"::);
}



/*****************************************************************************
*
*   Function name : Revision
*
*   Returns :       None
*
*   Parameters :    char input
*
*   Purpose :       Display the software revision
*
*****************************************************************************/
char Revision(char input)
{
    static char enter = 1;
    
    if(enter)
    {
        enter = 0;
        
		// mtA 
        LCD_putc(0, 'R'); // LCD_putc(0, 'R');
        LCD_putc(1, 'E'); // LCD_putc(1, 'E');
        LCD_putc(2, 'V'); // LCD_putc(2, 'V');
        // LCD_putc(3, ' ');
        LCD_putc(3, (SWHIGH + 0x30)); // LCD_putc(4, (SWHIGH + 0x30));       //SWHIGH/LOW are defined in "main.h"
        LCD_putc(4, (SWLOW + 0x30)); // LCD_putc(5, (SWLOW + 0x30));
		LCD_putc(5, (SWLOWLOW + 0x30)); // LCD_putc(5, (SWLOW + 0x30));
        LCD_putc(6, '\0');
		// mtE
        
        LCD_UpdateRequired(TRUE, 0);          
    }
    else if (input == KEY_PREV)
    {
        enter = 1;
        return ST_AVRBF;
    }
    
    return ST_AVRBF_REV;
}




/*****************************************************************************
*
*   Function name : OSCCAL_calibration
*
*   Returns :       None
*
*   Parameters :    None
*
*   Purpose :       Calibrate the internal OSCCAL byte, using the external 
*                   32,768 kHz crystal as reference
*
*****************************************************************************/
void OSCCAL_calibration(void)
{
    unsigned char calibrate = FALSE;
    int temp;
    unsigned char tempL;

    CLKPR = (1<<CLKPCE);        // set Clock Prescaler Change Enable
    // set prescaler = 8, Inter RC 8Mhz / 8 = 1Mhz
    CLKPR = (1<<CLKPS1) | (1<<CLKPS0);
    
    TIMSK2 = 0;             //disable OCIE2A and TOIE2

    ASSR = (1<<AS2);        //select asynchronous operation of timer2 (32,768kHz)
    
    OCR2A = 200;            // set timer2 compare value 

    TIMSK0 = 0;             // delete any interrupt sources
        
    TCCR1B = (1<<CS10);     // start timer1 with no prescaling
    TCCR2A = (1<<CS20);     // start timer2 with no prescaling

    while((ASSR & 0x01) | (ASSR & 0x04));       //wait for TCN2UB and TCR2UB to be cleared

    Delay(1000);    // wait for external crystal to stabilise
    
    while(!calibrate)
    {
        cli(); // mt __disable_interrupt();  // disable global interrupt
        
        TIFR1 = 0xFF;   // delete TIFR1 flags
        TIFR2 = 0xFF;   // delete TIFR2 flags
        
        TCNT1H = 0;     // clear timer1 counter
        TCNT1L = 0;
        TCNT2 = 0;      // clear timer2 counter
           
	// shc/mt while ( !(TIFR2 && (1<<OCF2A)) );   // wait for timer2 compareflag    
        while ( !(TIFR2 & (1<<OCF2A)) );   // wait for timer2 compareflag

        TCCR1B = 0; // stop timer1

        sei(); // __enable_interrupt();  // enable global interrupt
    
        // shc/mt if ( (TIFR1 && (1<<TOV1)) )
	if ( (TIFR1 & (1<<TOV1)) )
        {
            temp = 0xFFFF;      // if timer1 overflows, set the temp to 0xFFFF
        }
        else
        {   // read out the timer1 counter value
            tempL = TCNT1L;
            temp = TCNT1H;
            temp = (temp << 8);
            temp += tempL;
        }
    
        if (temp > 6250)
        {
            OSCCAL--;   // the internRC oscillator runs to fast, decrease the OSCCAL
        }
        else if (temp < 6120)
        {
            OSCCAL++;   // the internRC oscillator runs to slow, increase the OSCCAL
        }
        else
            calibrate = TRUE;   // the interRC is correct
    
        TCCR1B = (1<<CS10); // start timer1
    }
}
