//***************************************************************************
//
//  File........: sound.c
//
//  Author(s)...: ATMEL Norway
//
//  Target(s)...: ATmega169
//
//  Compiler....: GCC 3.3.1; avr-libc 1.0
//
//  Description.: AVR Butterfly sound routines
//
//  Revisions...: 1.0
//
//  YYYYMMDD - VER. - COMMENT                                       - SIGN.
//
//  20030116 - 1.0  - Created                                       - LHM
//  20031009          port to gcc/avr-libc                          - M.Thomas
//  20040123          added temp.-var.                              - n.n./mt
//
//***************************************************************************

#include <avr/io.h>
#include <avr/pgmspace.h>
#include <avr/interrupt.h>
#include <inttypes.h>
#include "main.h"
#include "sound.h"
#include "timer0.h"
#include "LCD_functions.h"


/******************************************************************************
*
*   A song is defined by a table of notes. The first byte sets the tempo. A 
*   high byte will give a low tempo, and opposite. Each tone consists of two 
*   bytes. The first gives the length of the tone, the other gives the frequency. 
*   The frequencies for each tone are defined in the "sound.h". Timer0 controls 
*   the tempo and the length of each tone, while Timer1 with PWM gives the 
*   frequency. The second last byte is a "0" which indicates the end, and the
*   very last byte makes the song loop if it's "1", and not loop if it's "0".
*
******************************************************************************/

// mt __flash char TEXT_SONG1[]       = "Fur Elise";
const char TEXT_SONG1[] PROGMEM      = "Fur Elise";

// __flash int FurElise[] =   
const int FurElise[] PROGMEM=   
        {
            3, 
            8,e2, 8,xd2, 8,e2, 8,xd2, 8,e2, 8,b1, 8,d2, 8,c2, 4,a1, 8,p, 
            8,c1, 8,e1, 8,a1, 4,b1, 8,p, 8,e1, 8,xg1, 8,b1, 4,c2, 8,p, 8,e1, 
            8,e2, 8,xd2, 8,e2, 8,xd2, 8,e2, 8,b1, 8,d2, 8,c2, 4,a1, 8,p, 8,c1, 
            8,e1, 8,a1, 4,b1, 8,p, 8,e1, 8,c2, 8,b1, 4,a1, 
            0, 1
        };


//__flash char TEXT_SONG2[]       = "Turkey march";
const char TEXT_SONG2[] PROGMEM  = "Turkey march";

//__flash int Mozart[] = 
const int Mozart[] PROGMEM = 
        {
            3, 
            16,xf1, 16,e1, 16,xd1, 16,e1, 4,g1, 16,a1, 16,g1, 16,xf1, 16,g1,
            4,b1, 16,c2, 16,b1, 16,xa1, 16,b1, 16,xf2, 16,e2, 16,xd2, 16,e2, 
            16,xf2, 16,e2, 16,xd2, 16,e2, 4,g2, 8,e2, 8,g2, 32,d2, 32,e2, 
            16,xf2, 8,e2, 8,d2, 8,e2, 32,d2, 32,e2, 16,xf2, 8,e2, 8,d2, 8,e2, 
            32,d2, 32,e2, 16,xf2, 8,e2, 8,d2, 8,xc2, 4,b1, 
            0, 1
        };

// mt song 3 & 4 where commented out by ATMEL - see their readme
// well, the gcc-geek wants all the songs ;-)
const char TEXT_SONG3[] PROGMEM      = "Minuet";

const int Minuet[] PROGMEM = 
        {
            2, 
            4,d2, 8,g1, 8,a1, 8,b1, 8,c2, 4,d2, 4,g1, 4,g1, 4,e2, 8,c2, 
            8,d2, 8,e2, 8,xf2, 4,g2, 4,g1, 4,g1, 4,c2, 8,d2, 8,c2, 8,b1, 
            8,a1, 4,b1, 8,c2, 8,b1, 8,a1, 8,g1, 4,xf1, 8,g1, 8,a1, 8,b1, 
            8,g1, 4,b1, 2,a1, 
            0, 1
        };


char TEXT_SONG4[] PROGMEM    = "Auld Lang Syne";

int AuldLangSyne[] PROGMEM = 
        {  
            3, 
            4,g2, 2,c3, 8,c3, 4,c3, 4,e3, 2,d3, 8,c3, 4,d3, 8,e3, 8,d3, 2,c3, 
            8,c3, 4,e3, 4,g3, 2,a3, 8,p, 4,a3, 2,g3, 8,e3, 4,e3, 4,c3, 2,d3, 
            8,c3, 4,d3, 8,e3, 8,d3, 2,c3, 8,a2, 4,a2, 4,g2, 2,c3, 4,p,
            0, 1
        };


//__flash char TEXT_SONG5[]      =   "Sirene1";
const char TEXT_SONG5[] PROGMEM =   "Sirene1";

//__flash int Sirene1[] = 
const int Sirene1[] PROGMEM = 
        {
            0,
            32,400, 32,397, 32,394, 32,391, 32,388, 32,385, 32,382, 32,379,
            32,376, 32,373, 32,370, 32,367, 32,364, 32,361, 32,358, 32,355,
            32,352, 32,349, 32,346, 32,343, 32,340, 32,337, 32,334, 32,331, 
            32,328, 32,325, 32,322, 32,319, 32,316, 32,313, 32,310, 32,307, 
            32,304, 32,301, 32,298, 32,298, 32,301, 32,304, 32,307, 32,310, 
            32,313, 32,316, 32,319, 32,322, 32,325, 32,328, 32,331, 32,334, 
            32,337, 32,340, 32,343, 32,346, 32,349, 32,352, 32,355, 32,358, 
            32,361, 32,364, 32,367, 32,370, 32,373, 32,376, 32,379, 32,382, 
            32,385, 32,388, 32,391, 32,394, 32,397, 32,400,
            0, 1
        };

//__flash char TEXT_SONG6[]      =   "Sirene2";
const char TEXT_SONG6[] PROGMEM =   "Sirene2";

//__flash int Sirene2[] = 
const int Sirene2[] PROGMEM = 
        {
            3, 
            4,c2, 4,g2, 
            0, 1
        };


//__flash char TEXT_SONG7[]      =   "Whistle";
const char TEXT_SONG7[] PROGMEM      =   "Whistle";

//__flash int Whistle[] = 
const int Whistle[] PROGMEM = 
        {
            0, 
            32,200, 32,195, 32,190, 32,185, 32,180, 32,175, 32,170, 32,165,  
            32,160, 32,155, 32,150, 32,145, 32,140, 32,135, 32,130, 32,125,              
            32,120, 32,115, 32,110, 32,105, 32,100, 8,p, 32,200, 32,195, 
            32,190, 32,185, 32,180, 32,175, 32,170, 32,165, 32,160, 32,155, 
            32,150, 32,145, 32,140, 32,135, 32,130, 32,125, 32,125, 32,130, 
            32,135, 32,140, 32,145, 32,150, 32,155, 32,160, 32,165, 32,170, 
            32,175, 32,180, 32,185, 32,190, 32,195, 32,200, 
            0, 0
        };


// pointer-array with pointers to the song arrays
// mt: __flash int __flash *Songs[]    = { FurElise, Mozart, /*Minuet, AuldLangSyne,*/ Sirene1, Sirene2, Whistle, 0};
const int *Songs[] PROGMEM   = { FurElise, Mozart, Minuet, AuldLangSyne, Sirene1, Sirene2, Whistle, 0};

//mt: __flash char __flash *TEXT_SONG_TBL[]    = { TEXT_SONG1, TEXT_SONG2, /*TEXT_SONG3, TEXT_SONG4,*/TEXT_SONG5, TEXT_SONG6, TEXT_SONG7, 0};
PGM_P TEXT_SONG_TBL[] PROGMEM   = { TEXT_SONG1, TEXT_SONG2, TEXT_SONG3, TEXT_SONG4, TEXT_SONG5, TEXT_SONG6, TEXT_SONG7, 0};
// const char *TEXT_SONG_TBL[]    = { TEXT_SONG1, TEXT_SONG2, TEXT_SONG3, TEXT_SONG4, TEXT_SONG5, TEXT_SONG6, TEXT_SONG7, 0};

//__flash char PLAYING[]          = "PLAYING";
const char PLAYING[] PROGMEM   = "PLAYING";

// mt: int __flash *pSong;     // pointer to the different songs in flash
const int *pSong;	// mt point to a ram location (pointer array Songs)

extern char gPlaying;      // global variable from "main.c"
static char Volume = 80;
static char Duration = 0;
static char Tone = 0;
static char Tempo;


/*****************************************************************************
*
*   Function name : Sound_Init
*
*   Returns :       None
*
*   Parameters :    None
*
*   Purpose :       Set up Timer1 with PWM
*
*****************************************************************************/
void Sound_Init(void)
{
    TCCR1A = (1<<COM1A1);// | (1<<COM1A0); // Set OC1A when upcounting, clear when downcounting
    TCCR1B = (1<<WGM13);        // Phase/Freq-correct PWM, top value = ICR1
    
    sbiBF(TCCR1B, CS10);             // start Timer1, prescaler(1)    
    
	// mtA
    OCR1AH = 0; //OCR1AH = 0;     // Set a initial value in the OCR1A-register
    OCR1AL = Volume; //OCR1AL = Volume;     // This will adjust the volume on the buzzer, lower value => higher volume
    // mtE
    
    Timer0_RegisterCallbackFunction(Play_Tune);     // Timer/Counter0 keeps the right beat
}

/*****************************************************************************
*
*   Function name : SelectSound
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Select song/tune
*
*****************************************************************************/
// mt inserted local helper to save some flash-space
void showSongName(unsigned char song)
{
	LCD_puts_f((PGM_P)pgm_read_word(&TEXT_SONG_TBL[song]), 1);  // mt   // Set up the a song in the LCD
}

char SelectSound(char input)
{
    static char enter = 1;
    // mt static char song = 0;
	static uint8_t song = 0;
    
    
    if (enter)
    {
        enter = 0;
        
        // mt LCD_puts_f((PGM_P)pgm_read_word(&TEXT_SONG_TBL[song]), 1);  // mt   // Set up the a song in the LCD
		showSongName(song);

        // mt pSong = Songs[song];            // point to this song             
		pSong=(int*)pgm_read_word(&Songs[song]); // looks too complicated...

    }      
        
    if (input == KEY_PLUS)  // shift to next song
    {
        if(!song)       // wrap around the table
        {
            for(song=1; pgm_read_word(&TEXT_SONG_TBL[song]); song++){}; // mt
            
            song--;
        }    
        else
            song--;

        // mt LCD_puts_f((PGM_P)pgm_read_word(&TEXT_SONG_TBL[song]), 1); // mt
		showSongName(song);
        
        // mt pSong = Songs[song];
		pSong=(int*)pgm_read_word(&Songs[song]); 

    }    
    else if (input == KEY_MINUS)    // shift to next song
    {
        song++;

        if( !(pgm_read_word(&TEXT_SONG_TBL[song])) )       // wrap around the table
            song = 0;
        
        // mt LCD_puts_f((PGM_P)pgm_read_word(&TEXT_SONG_TBL[song]), 1);
		showSongName(song);
        
        // mt pSong = Songs[song];
		pSong=(int*)pgm_read_word(&Songs[song]);
    }  
    else if(input == KEY_ENTER)     // start playing
    {
        enter = 1;
        return ST_MUSIC_PLAY;
    }
    else if (input == KEY_PREV)
    {
        enter = 1;
        return ST_MUSIC;
    }    
        
    return ST_MUSIC_SELECT;
}

/*****************************************************************************
*
*   Function name : Sound
*
*   Returns :       char ST_state (to the state-machine)
*
*   Parameters :    char input (from joystick)
*
*   Purpose :       Start/stop timers, adjust volume 
*
*****************************************************************************/
char Sound(char input)
{
    static char enter = 1;

    if (enter)
    {
        enter = 0;
        Tone = 0;
        Sound_Init();           // start playing
        LCD_puts_f(PLAYING, 1);
        gPlaying = TRUE;
    }        
    else if (!gPlaying)
    {
        Timer0_RemoveCallbackFunction(Play_Tune);
        TCCR1A = 0;
        TCCR1B = 0;
        enter = 1;
        return ST_MUSIC_SELECT;
    }
    
    if (input == KEY_PLUS)          // increase the volum
    {
        if(Volume >= 80)
            Volume = 80;
        else
            Volume += 5;
        // mtA       
		OCR1AH = 0;	// OCR1AH = 0;
        OCR1AL = Volume;	// OCR1AL = Volume;
		// mtE
    }
    else if (input == KEY_MINUS)    // decrease the volum
    {
        if(Volume < 11)
            Volume = 6;
        else
            Volume -= 5;   
        
		OCR1AH = 0;
        OCR1AL = Volume;
    }         
    
    if (input == KEY_ENTER)         // start/stop playing
    {   
        if (gPlaying)
        {
            gPlaying = FALSE;
            cbiBF(TCCR1B, 0);                     // stop Playing
            Timer0_RemoveCallbackFunction(Play_Tune);
            TCCR1A = 0;
            TCCR1B = 0;
            sbiBF(PORTB, 5);              // set OC1A high
            enter = 1;
            return ST_MUSIC_SELECT;
        }
        else
        {   
            Duration = 0;                       // start Playing
            Tone = 1;
            Sound_Init();
            LCD_puts_f(PLAYING, 1);            
            gPlaying = TRUE;
        }
    }
    else if (input == KEY_PREV)
    {
        gPlaying = FALSE;
        cbiBF(TCCR1B, 0);                     // stop Playing
        Timer0_RemoveCallbackFunction(Play_Tune);
        TCCR1A = 0;
        TCCR1B = 0;
        sbiBF(PORTB, 5);              // set OC1A high
        enter = 1;
        return ST_MUSIC_SELECT;
    }
        
    return ST_MUSIC_PLAY;
}
/*****************************************************************************
*
*   Function name : Play_Tune
*
*   Returns :       None
*
*   Parameters :    None
*
*   Purpose :       Plays the song
*
*****************************************************************************/
void Play_Tune(void)
{
    unsigned int temp_tone;	// mt 200301
    int temp_hi;
    
    char loop;
    
    if(!Tone)
    {
        Duration = 0;   
        // mt Tempo = *(pSong + 0);
        Tempo = (uint8_t)pgm_read_word(pSong + 0);
        Tone = 1;   //Start the song from the beginning
    }
    
    if(!Tempo)
    {
        if(Duration)        // Check if the lenght of the tone has "expired"
        {   
            Duration--;
        }
        // mt: else if(*(pSong + Tone))    // If not the end of the song
		else if(pgm_read_word(pSong + Tone))  // If not the end of the song
        {
            // mt: Duration = ( DURATION_SEED / *(pSong + Tone) );  // store the duration
            Duration = ( DURATION_SEED / pgm_read_word(pSong + Tone) );  // store the duration
        
			Tone++;                     // point to the next tone in the Song-table        

			temp_tone=pgm_read_word(pSong + Tone); // mt 200301
			// mt: if( (*(pSong + Tone) == p) | (*(pSong + Tone) == P) ) // if pause
			// if( (pgm_read_word(pSong + Tone) == p) | (pgm_read_word(pSong + Tone) == P) ) // if pause
            if( (temp_tone == p) | (temp_tone == P) ) // if pause
                cbiBF(TCCR1B, CS10);             // stop Timer1, prescaler(1)    
            else 
                sbiBF(TCCR1B, CS10);             // start Timer1, prescaler(1)  
                
            cli(); // mt __disable_interrupt();
            
            // mt temp_hi = *(pSong + Tone);      // read out the PWM-value
			// temp_hi = pgm_read_word(pSong + Tone);      // read out the PWM-value
			temp_hi = temp_tone;			// mt 200301
            temp_hi >>= 8;                  // move integer 8 bits to the rigth
                
            TCNT1H = 0;                     // reset TCNT1H/L
            TCNT1L = 0;
            
            ICR1H = temp_hi;                // load ICR1H/L
            // mt: ICR1L = *(pSong + Tone);        
			// ICR1L = pgm_read_word(pSong + Tone);
			ICR1L = temp_tone;
            
            sei(); // mt: __enable_interrupt();
            
            Tone++;                     // point to the next tone in the Song-table
        }
        else    // the end of song
        {
            Tone++;         // point to the next tone in the Song-table        
            
            // mt: loop = *(pSong + Tone); // get the byte that tells if the song should loop or not
			loop = (uint8_t)pgm_read_word(pSong + Tone); // get the byte that tells if the song should loop or not
            
            if( loop )  
            {
                Tone = 1;
            }
            else        // if not looping the song
            {
                Tone = 0;
                gPlaying = FALSE;
                cbiBF(TCCR1B, 0);                     // stop Playing
                Timer0_RemoveCallbackFunction(Play_Tune);
                TCCR1A = 0;
                TCCR1B = 0;
                sbiBF(PORTB, 5);              // set OC1A high
            }
        }
        
        // mt: Tempo = *(pSong + 0);
		Tempo = (uint8_t)pgm_read_word(pSong + 0);
    }
    else
        Tempo--;
 
}    

//mtA
void PlayClick(void)
{
        unsigned char i;
        for (i = 0; i < 10; i++) {
                sbiBF(PORTB, 5);
                Delay(1);
                cbiBF(PORTB, 5);
                Delay(1);
        }
}
//mtE