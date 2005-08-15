// Button.h

#ifdef M162
#define PINA_MASK ((1<<PINA0)|(1<<PINA1)|(1<<PINA2)|(1<<PINA3)|(1<<PINA4))
#else
#define PINB_MASK ((1<<PINB4)|(1<<PINB6)|(1<<PINB7))
#define PINE_MASK ((1<<PINE2)|(1<<PINE3))
#endif

#ifdef M162
#define BUTTON_A    0   // NORTH
#define BUTTON_B    1   // EAST
#define BUTTON_C    2   // WEST
#define BUTTON_D    3   // SOUTH
#define BUTTON_O    4   // PUSH
#else
#define BUTTON_A    6   // UP
#define BUTTON_B    7   // DOWN
#define BUTTON_C    2   // LEFT
#define BUTTON_D    3   // RIGHT
#define BUTTON_O    4   // PUSH
#endif

void PinChangeInterrupt(void);
void Button_Init(void);
char getkey(void);
char ButtonBouncing(void);

