//***************************************************************************
//
//  File........: ADC.c
//
//  Author(s)...: ATMEL Norway
//
//  Target(s)...: ATmega169
//
//  Compiler....: AVR-GCC 3.3.1; avr-libc 1.0
//
//  Description.: AVR Butterfly ADC routines
//
//  Revisions...: 1.0
//
//  YYYYMMDD - VER. - COMMENT                                       - SIGN.
//
//  20030116 - 1.0  - Created                                       - LHM
//  20031009          port to avr-gcc/avr-libc                      - M.Thomas
//
//***************************************************************************

//mtA
//#include <inavr.h>
//#include "iom169.h"
#include <avr/io.h>
#include <avr/pgmspace.h>
#include "pgmspacehlp.h"
//mtE

#include "main.h"
#include "ADC.h"
#include "BCD.h"
#include "LCD_functions.h"
#include "timer0.h"

// mt __flash int TEMP_Celcius_pos[] =    // Positive Celcius temperatures (ADC-value)
const int TEMP_Celsius_pos[] PROGMEM =    // Positive Celsius temperatures (ADC-value)
        {                           // from 0 to 60 degrees
            806,796,786,775,765,754,743,732,720,709,697,685,673,661,649,
            636,624,611,599,586,574,562,549,537,524,512,500,488,476,464,
            452,440,429,418,406,396,385,374,364,354,344,334,324,315,306,
            297,288,279,271,263,255,247,240,233,225,219,212,205,199,193,
            187,
        };

// mt __flash int TEMP_Celcius_neg[] =    // Negative Celcius temperatures (ADC-value)
const int TEMP_Celsius_neg[] PROGMEM =    // Negative Celsius temperatures (ADC-value)
        {                           // from -1 to -15 degrees
            815,825,834,843,851,860,868,876,883,891,898,904,911,917,923,
        };

// mt __flash int TEMP_Farenheit_pos[] =  // Positive Farenheit temperatures (ADC-value)
const int TEMP_Fahrenheit_pos[] PROGMEM =  // Positive Fahrenheit temperatures (ADC-value)
        {                           // from 0 to 140 degrees
		    938, 935, 932, 929, 926, 923, 920, 916, 913, 909, 906, 902, 898, 
    		894, 891, 887, 882, 878, 874, 870, 865, 861, 856, 851, 847, 842, 
    		837, 832, 827, 822, 816, 811, 806, 800, 795, 789, 783, 778, 772, 
    		766, 760, 754, 748, 742, 735, 729, 723, 716, 710, 703, 697, 690, 
    		684, 677, 670, 663, 657, 650, 643, 636, 629, 622, 616, 609, 602, 
    		595, 588, 581, 574, 567, 560, 553, 546, 539, 533, 526, 519, 512, 
    		505, 498, 492, 485, 478, 472, 465, 459, 452, 446, 439, 433, 426, 
    		420, 414, 408, 402, 396, 390, 384, 378, 372, 366, 360, 355, 349, 
    		344, 338, 333, 327, 322, 317, 312, 307, 302, 297, 292, 287, 282, 
    		277, 273, 268, 264, 259, 255, 251, 246, 242, 238, 234, 230, 226, 
    		222, 219, 215, 211, 207, 204, 200, 197, 194, 190, 187, 
        };

        
// mt __flash int LIGHT_ADC[] = // Table used to find the Vref, when using the voltage-reading function 
const int LIGHT_ADC[] PROGMEM = // Table used to find the Vref, when using the voltage-reading function 
        { 
            0x35,0x60,0x80,0x0B0,0x11D,0x13D,0x15A,0x17A,0x197,0x1B9,0x1DA,
            0x1F9,0x216,0x240,0x26D,0x282,0x2A2,0x2EF,0x332,0x3B0,0x3F2
        };
        
//mt __flash float LIGHT_VOLTAGE[] = // Vref table correspondent to the LIGHT_ADC[] table
const float LIGHT_VOLTAGE[] PROGMEM = // Vref table correspondent to the LIGHT_ADC[] table
        {
            2.818,2.820,2.824,2.827,2.832,2.835,2.839,2.841,2.843,2.847,2.850,
            2.853,2.857,2.863,2.867,2.870,2.874,2.882,2.893,2.917,2.939
        };


float Vref = 2.900; // initial value
char degree = CELSIUS; // char degree = CELCIUS;


/*****************************************************************************
*
*   Function name : ADC_init
*
*   Returns :       None
*
*   Parameters :    char input
*
*   Purpose :       Initialize the ADC with the selected ADC-channel
*
*****************************************************************************/
void ADC_init(char input)
{
  
    ADMUX = input;    // external AREF and ADCx
    
    ADCSRA = (1<<ADEN) | (1<<ADPS1) | (1<<ADPS0);    // set ADC prescaler to , 1MHz / 8 = 125kHz    

    input = ADC_read();        // dummy 
}


/*****************************************************************************
*
*   Function name : ADC_read
*
*   Returns :       int ADC
*
*   Parameters :    None
*
*   Purpose :       Do a Analog to Digital Conversion
*
*****************************************************************************/
int ADC_read(void)
{
    char i;
    int ADC_temp;
	// mt int ADC = 0 ;
	int ADCr = 0;
    
    // To save power, the voltage over the LDR and the NTC is turned off when not used
    // This is done by controlling the voltage from a I/O-pin (PORTF3)
    sbiBF(PORTF, PF3); // mt sbi(PORTF, PORTF3);     // Enable the VCP (VC-peripheral)
    sbiBF(DDRF, DDF3); // sbi(DDRF, PORTF3);        

    sbiBF(ADCSRA, ADEN);     // Enable the ADC

    //do a dummy readout first
    ADCSRA |= (1<<ADSC);        // do single conversion
    while(!(ADCSRA & 0x10));    // wait for conversion done, ADIF flag active
        
    for(i=0;i<8;i++)            // do the ADC conversion 8 times for better accuracy 
    {
        ADCSRA |= (1<<ADSC);        // do single conversion
        while(!(ADCSRA & 0x10));    // wait for conversion done, ADIF flag active
        
        ADC_temp = ADCL;            // read out ADCL register
        ADC_temp += (ADCH << 8);    // read out ADCH register        

        ADCr += ADC_temp;      // accumulate result (8 samples) for later averaging
    }

    ADCr = ADCr >> 3;     // average the 8 samples
        
    cbiBF(PORTF,PF3); // mt cbi(PORTF, PORTF3);     // disable the VCP
    cbiBF(DDRF,DDF3); // mt cbi(DDRF, PORTF3);  
    
    cbiBF(ADCSRA, ADEN);      // disable the ADC

    return ADCr;
}


/*****************************************************************************
*
*   Function name : ADC_periphery
*
*   Returns :       None
*
*   Parameters :    None
*
*   Purpose :       Calculates the Temperature/Voltage/Ligth from the ADC_read
*                   and puts it out on the LCD.
*
*****************************************************************************/
void ADC_periphery(void)
{
    int ADCresult = 0;
    int ADCresult_temp = 0;
    int Temp_int;
    char Temp;
    unsigned char i = 0;
    char TL;
    char TH;    
    char VL;

    float V_ADC;
    char VoltageHB;
    char VoltageLB;
    
    ADCresult = ADC_read();         // Find the ADC value
     
    if( ADMUX == TEMPERATURE_SENSOR )
    {
        if(degree == CELSIUS)
        {
            if(ADCresult > 810)         // If it's a negtive temperature
            {    
                for (i=0; i<=25; i++)   // Find the temperature
                {
                    // mt if (ADCresult <= TEMP_Celcius_neg[i])
					if (ADCresult <= pgm_read_word(&TEMP_Celsius_neg[i]))
                    {
                        break;
                    }
                }
                
                LCD_putc(1, '-');       // Put a minus sign in front of the temperature
            }
            else if (ADCresult < 800)   // If it's a positive temperature
            {
                for (i=0; i<100; i++)  
                {
                    // mt if (ADCresult >= TEMP_Celcius_pos[i])
					if (ADCresult >= pgm_read_word(&TEMP_Celsius_pos[i]))
                    {
                        break;
                    }
                }        
            
                LCD_putc(1, '+');       // Put a plus sign in front of the temperature
            }
            else                        //If the temperature is zero degrees
            {
                i = 0;
                LCD_putc(1, ' ');
            }
            
            Temp = CHAR2BCD2(i);        // Convert from char to bin
    
            TL = (Temp & 0x0F) + '0';   // Find the low-byte
            TH = (Temp >> 4) + '0';     // Find the high-byte
            
            LCD_putc(0, ' ');
            //LCD character 1 is allready written to
            LCD_putc(2, TH);
            LCD_putc(3, TL);
            LCD_putc(4, '*');
            LCD_putc(5, 'C');
            LCD_putc(6, '\0');
        }
        else if (degree == FAHRENHEIT)
        {
            for (i=0; i<=141; i++)   // Find the temperature
            {
                // mt if (ADCresult > TEMP_Farenheit_pos[i])
				if (ADCresult > pgm_read_word(&TEMP_Fahrenheit_pos[i]))
                {
                    break;
                }
            }        
        
            Temp_int = CHAR2BCD3(i);
        
            if (i > 99) // if there are three digits
            {
                LCD_putc(0, '+');
                TH = (Temp_int >> 8) + '0';   // Find the high-byte
                LCD_putc(1, TH);
            }
            else    // if only two digits
            {
                LCD_putc(0, ' ');
                LCD_putc(1, '+');
            }
            
            TL = (Temp_int & 0x0F) + '0';   // Find the low-byte
            TH = ( (Temp_int >> 4) & 0x0F ) + '0';     // Find the high-byte                

            LCD_putc(2, TH);
            LCD_putc(3, TL);
            LCD_putc(4, '*');
            LCD_putc(5, 'F');
            LCD_putc(6, '\0');
        
        }

//        Can't set LCD_UpdateRequired = TRUE here cause we are inside the Timer0 interrupt
//        LCD_UpdateRequired(TRUE, 0);        
           
    }
    else if( ADMUX == VOLTAGE_SENSOR )
    {
       
//  Do a Light-measurement first to determine the Vref, 
//  because the LDR affects the Vref 

        ADCresult_temp = ADCresult;     // Store the ADCresult from the voltage reading
          
        ADC_init(LIGHT_SENSOR);         // Init the ADC to measure light
            
        ADCresult = ADC_read();         // Read the light value

        // Find Vref
        for (i=0; i<=22; i++)
        {
            // mt if (ADCresult <= LIGHT_ADC[i])
			if (ADCresult <= pgm_read_word(&LIGHT_ADC[i]))
            {
                break;
            }
        }
        if(!i)              // if it's very bright
            Vref = 2.815;
        else if(i > 21)
            Vref = 2.942;   // if it's totally dark
        else
            // mt: Vref = LIGHT_VOLTAGE[i];   
			// mt using helper from pgmspacehlp.h - new code
			Vref = pgm_read_float_hlp(&LIGHT_VOLTAGE[i]);
          
        ADMUX = VOLTAGE_SENSOR;     
        ADCresult = ADCresult_temp; // Get the ADCresult from the voltage reading

// Light-measurement finished 
            
                    
        V_ADC = ( ADCresult * Vref ) / 1024; // Calculate the voltage
           
        V_ADC = ( V_ADC * 6 );      // Multiply by 6 cause of the voltage division
            
        VoltageHB = V_ADC;              // Store the high-byte
        V_ADC = ( V_ADC - VoltageHB );
        VoltageLB = ( V_ADC * 100 );    // Store the low-byte
     
        Temp = CHAR2BCD2(VoltageHB);    // Convert from char to bin
    
        TL = (Temp & 0x0F) + '0';
        TH = (Temp >> 4) + '0';
            
        Temp = CHAR2BCD2(VoltageLB);    // Convert from char to bin
   
        VL = (Temp >> 4) + '0';

        LCD_putc(0, ' ');
        LCD_putc(1, ' ');
        LCD_putc(2, ' ');
        LCD_putc(3, TL);
        LCD_putc(4, 'V');
        LCD_putc(5, VL);
        LCD_putc(6, '\0');
        
//        Can't set LCD_UpdateRequired = TRUE here cause we are inside the Timer0 interrupt
//        LCD_UpdateRequired(TRUE, 0);        
                             
    }
    else if( ADMUX == LIGHT_SENSOR )
    {
        // The relation between ADC-value and lux is yet to be found, 
        // for now the ADC-value is presented on the LCD
        
        VoltageHB = CHAR2BCD2(ADCH);    // Convert from char to bin

        Temp = ADCL;                
    
        TL = (Temp & 0x0F) + '0';       
        if(TL > '9')        // if the hex-value is over 9, add 7 in order to go 
            TL += 7;        // jump to the character in the ASCII-table
                
        TH = (Temp >> 4) + '0';
        if(TH > '9')        // if the hex-value is over 9, add 7 in order to go 
            TH += 7;        // jump to the character in the ASCII-table
            
        LCD_putc(0, 'A');
        LCD_putc(1, 'D');
        LCD_putc(2, 'C');
        LCD_putc(3, (ADCH + 0x30));
        LCD_putc(4, TH);
        LCD_putc(5, TL);
        LCD_putc(6, '\0');

//        Can't set LCD_UpdateRequired = TRUE here cause we are inside the Timer0 interrupt
//        LCD_UpdateRequired(TRUE, 0);        

    }
}


/*****************************************************************************
*
*   Function name : TemperatureFunc
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Enable or disable temperature measurements
*
*****************************************************************************/
char TemperatureFunc(char input)
{
    static char enter = 1;
    
    if (enter)
    {
        enter = 0;
        
        ADC_init(TEMPERATURE_SENSOR);       // Init the ADC

        // Enable auto-run of the ADC_perphery every 10ms 
        // (it will actually be more than 10ms cause of the SLEEP)
        Timer0_RegisterCallbackFunction(ADC_periphery); 
    }
    else
        LCD_UpdateRequired(TRUE, 0);        // New data to be presented
    
    if (input == KEY_PREV)
    {
        // Disable the auto-run of the ADC_periphery
        Timer0_RemoveCallbackFunction(ADC_periphery);
        
        enter = 1;  // Set enter to 1 before leaving the TemperatureFunc
        
        return ST_TEMPERATURE;
    }
    else if (input == KEY_PLUS)
    {   
        if (degree == FAHRENHEIT)
            degree = CELSIUS;
        else
            degree = FAHRENHEIT;
    }
    else if (input == KEY_MINUS)
    {
        if (degree == FAHRENHEIT)
            degree = CELSIUS;
        else
            degree = FAHRENHEIT;
    }
    
    return ST_TEMPERATURE_FUNC;        
}


/*****************************************************************************
*
*   Function name : VoltageFunc
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Enable or disable voltage measurements
*
*****************************************************************************/
char VoltageFunc(char input)
{
    static char enter = 1;
    
    if (enter)
    {
        enter = 0;

        ADC_init(VOLTAGE_SENSOR);       // Init the ADC
        
        // Enable auto-run of the ADC_perphery every 10ms 
        // (it will actually be more than 10ms cause of the SLEEP)        
        Timer0_RegisterCallbackFunction(ADC_periphery);        
    }
    else
        LCD_UpdateRequired(TRUE, 0); 

    if (input == KEY_PREV)
    {
        // Disable the auto-run of the ADC_periphery        
        Timer0_RemoveCallbackFunction(ADC_periphery);
        
        enter = 1;  // Set enter to 1 before leaving the TemperatureFunc
        
        return ST_VOLTAGE;
    }
    else
        return ST_VOLTAGE_FUNC;    
}    


/*****************************************************************************
*
*   Function name : LightFunc
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Enable or disable light measurements
*
*****************************************************************************/
char LightFunc(char input)
{
    static char enter = 1;
    
    if (enter)
    {
    
        enter = 0;
        
        ADC_init(LIGHT_SENSOR);     // Init the ADC
        
        // Enable auto-run of the ADC_perphery every 10ms 
        // (it will actually be more than 10ms cause of the SLEEP)  
        Timer0_RegisterCallbackFunction(ADC_periphery);        
    }
    else
        LCD_UpdateRequired(TRUE, 0); 

    if (input == KEY_PREV)
    {
        // Disable the auto-run of the ADC_periphery      
        Timer0_RemoveCallbackFunction(ADC_periphery);
        
        enter = 1;  // Set enter to 1 before leaving the TemperatureFunc
    
        return ST_LIGHT;
    }
    else
        return ST_LIGHT_FUNC;    
}    
