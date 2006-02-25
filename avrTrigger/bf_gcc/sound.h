void Sound_Init(void);
char SelectSound(char);
char Sound(char);
void Play_Tune(void);

void PlayClick(void); // mt

// global variable to prevent entering power save, when playing.
extern volatile char gPlaying; 

#define DURATION_SEED 32  

/*****************************************************************************
*
*   Tone definition. Each tone are set up with a value which will give the 
*   right frequency when applied to a 16-bits timer with PWM. These values are based
*   on a CLKcpu running @ 1Mhz.
*
*   First find the frequency for all tones. 
*
*   Formula:    ToneX = Bf * 2^(ToneX/12)
*   
*   ToneX: the actual tone, e.g. C0 = 3
*   Bf: Basefrequency = 220Hz (A)
*
*       
*   E.g: For tone C0 this would be:     C0 = 220 * 2^(3/12)
*                                       C0 = 261,6256...
*
*   Now we must find which value to put in a 16-bits timer with PWM to achieve 
*   this frequency
*
*   Formula:    Timer value = 1Mhz / ToneHz / 2
*
*   E.g: For tone C0 this would be:     Timer value = 1000000 / 261,6256... / 2
*                                       Timer value = 1911
*
*   Set up a 16-bits timer to run at Phase/Freq-correct PWM, top value = ICR1,
*   set OC1A when upcounting, clear when downcounting.
*   
*****************************************************************************/

#define A   2273        // tone 0
#define xA  2145        // tone 1
#define Ax  2145        // tone 1
#define B   2025        // tone 2
#define C0  1911        // tone 3
#define xC0 1804        // ...
#define Cx0 1804
#define D0  1703
#define xD0 1607
#define Dx0 1607
#define E0  1517
#define F0  1432
#define xF0 1351
#define Fx0 1351
#define G0  1275
#define xG0 1204
#define Gx0 1204
#define A0  1136
#define xA0 1073
#define Ax0 1073
#define B0  1012
#define C1  956
#define xC1 902
#define Cx1 902
#define D1  851
#define xD1 804
#define Dx1 804
#define E1  758
#define F1  716
#define xF1 676
#define Fx1 676
#define G1  638
#define xG1 602
#define Gx1 602
#define A1  568
#define xA1 536
#define Ax1 536
#define B1  506
#define C2  478
#define xC2 451
#define Cx2 451
#define D2  426
#define xD2 402
#define Dx2 402
#define E2  379
#define F2  356
#define xF2 338
#define Fx2 338
#define G2  319
#define xG2 301
#define Gx2 301
#define A2  284
#define xA2 268
#define Ax2 268
#define B2  253
#define C3  239
#define xC3 225
#define Cx3 225
#define D3  213
#define xD3 201
#define Dx3 201
#define E3  190
#define F3  179
#define xF3 169
#define Fx3 169
#define G3  159
#define xG3 150
#define Gx3 150
#define A3  142
#define xA3 134
#define Ax3 134
#define B3  127
#define C4  119


#define P   1           // pause



/******************************************************************************
*
*   The tone definitions are duplicated to accept both upper and lower case
*
******************************************************************************/

#define a   2273        // tone 0
#define xa  2145        // tone 1
#define ax  2145        // tone 1
#define b   2024        // tone 2
#define c0  1911        // tone 3
#define xc0 1804        // ...
#define cx0 1804
#define d0  1703
#define xd0 1607
#define dx0 1607
#define e0  1517
#define f0  1432
#define xf0 1351
#define fx0 1351
#define g0  1275
#define xg0 1204
#define gx0 1204
#define a0  1136
#define xa0 1073
#define ax0 1073
#define b0  1012
#define c1  956
#define xc1 902
#define cx1 902
#define d1  851
#define xd1 804
#define dx1 804
#define e1  758
#define f1  716
#define xf1 676
#define fx1 676
#define g1  638
#define xg1 602
#define gx1 602
#define a1  568
#define xa1 536
#define ax1 536
#define b1  506
#define c2  478
#define xc2 451
#define cx2 451
#define d2  426
#define xd2 402
#define dx2 402
#define e2  379
#define f2  356
#define xf2 338
#define fx2 338
#define g2  319
#define xg2 301
#define gx2 301
#define a2  284
#define xa2 268
#define ax2 268
#define b2  253
#define c3  239
#define xc3 225
#define cx3 225
#define d3  213
#define xd3 201
#define dx3 201
#define e3  190
#define f3  179
#define xf3 169
#define fx3 169
#define g3  159
#define xg3 150
#define gx3 150
#define a3  142
#define xa3 134
#define ax3 134
#define b3  127
#define c4  119



#define p   1
