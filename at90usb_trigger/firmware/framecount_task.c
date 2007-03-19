#include "config.h"
#include "framecount_task.h"

int64_t framecount_A=0;
volatile uint8_t do_increment_A_count=0;

void increment_framecount_A() {
  do_increment_A_count++;
}

void get_framecount_A(int64_t* result) {
  framecount_task(); // make sure no pending counts are around
  *result = framecount_A;
}

void framecount_task_init(void) {
  do_increment_A_count=0;
  framecount_A=0;
}

void framecount_task(void) {
  while (do_increment_A_count) {
    framecount_A++;
    do_increment_A_count--;
  }
}
