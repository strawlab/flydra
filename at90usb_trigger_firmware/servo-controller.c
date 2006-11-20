//_____  I N C L U D E S ___________________________________________________

#include "config.h"
#include "scheduler.h"
#include "wdt_drv.h"



//_____ M A C R O S ________________________________________________________

//_____ D E F I N I T I O N S ______________________________________________



void main(void)
{
  //  CLKPR = 0x00; // Full crystal speed
  Wdt_off();
  scheduler();
}


