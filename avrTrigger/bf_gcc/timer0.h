
// Typedef for Timer callback function 
typedef void (*TIMER_CALLBACK_FUNC) (void);

#define TIMER0_NUM_CALLBACKS        4
#define TIMER0_NUM_COUNTDOWNTIMERS  4

void Timer0_Init(void);

BOOL Timer0_RegisterCallbackFunction(TIMER_CALLBACK_FUNC pFunc);
BOOL Timer0_RemoveCallbackFunction(TIMER_CALLBACK_FUNC pFunc);


//mt 
// char Timer0_AllocateCountdownTimer();
char Timer0_AllocateCountdownTimer(void);
char Timer0_GetCountdownTimer(char timer);
void Timer0_SetCountdownTimer(char timer, char value);
void Timer0_ReleaseCountdownTimer(char timer);
