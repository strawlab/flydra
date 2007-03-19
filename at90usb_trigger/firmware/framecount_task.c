#include "config.h"

/*
#include "conf_usb.h"
#include "hid_task.h"
#include "stk_525.h"
#include "usb_drv.h"
#include "usb_descriptors.h"
#include "usb_standard_request.h"
#include "usb_specific_request.h"
#include "adc_drv.h"
*/

#include "framecount_task.h"

int64_t framecount_A=0;
volatile uint8_t do_increment_A_count=0; // this is like Python's threading.Event

void increment_framecount_A() {
  do_increment_A_count++;
}

void get_framecount_A(int64_t* result) {
  framecount_task(); // make sure no pending counts are around
  *result = framecount_A;
}

void framecount_task_init(void) {

}

void framecount_task(void) {
  while (do_increment_A_count) {
    framecount_A++;
    do_increment_A_count--;
  }
}
