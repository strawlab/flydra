#include "adc_sampling.h"
#include "adc_drv.h"

volatile uint8 adc_sample_status=0; // 0 = not ready, already read; 1 = ready to read
volatile uint8 adc_sample_skipped_error=0;

void init_ADC_sampling(void) {
   init_adc();
   Select_adc_channel(1); // channel 0 is thermistor
   Enable_adc_it(); // enable interrupt

   // XXX check the sample clock to ensure it's between 50-200 kHz
}

void start_ADC_sample(void) {
  // xxx
  
  // If (previous) ADC sample still ready (not ready yet), note that a
  // sample has been skipped.
  if (adc_sample_status==1)
    adc_sample_skipped_error=1;

  // Mark the (current/previous) sample invalid.
  adc_sample_status=0;

  // trigger an ADC read. an interrupt happens when it's done.
  Start_conv();
}

ISR(ADC_vect) {
  // mark (current) sample as valid but unread
  adc_sample_status=1;
}
