//_____  I N C L U D E S ___________________________________________________

#include "config.h"
#include "scheduler.h"
#include "wdt_drv.h"



//_____ M A C R O S ________________________________________________________

//_____ D E F I N I T I O N S ______________________________________________

int main(void)
{
 // Set FOSC to 8 MHz
 CLKPR = 0x80; // CLKPCE Clock Prescaler Change Enable set
 CLKPR = 0x00; // Clock Prescaler Select Bits set
 Wdt_off();
 scheduler();
 return 0;
}
