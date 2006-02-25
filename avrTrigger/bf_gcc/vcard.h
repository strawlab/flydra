char RS232(char input);
char vCard(char input);
char EnterName(char input);

#define STRLENGHT    25
// mt: moved to this location from eeprom.h:
#define EEPROM_START 0x100

// global variable to prevent entering power save, when using the UART
extern char gUART;
