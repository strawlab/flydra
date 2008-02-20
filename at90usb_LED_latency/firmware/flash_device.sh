#!/bin/bash
flydra_trigger_enter_dfu_mode --ignore-version-mismatch && \
sleep 3 && \
sudo dfu-programmer at90usb1287 erase && \
sudo dfu-programmer at90usb1287 flash trigger_control.hex && \
sudo dfu-programmer at90usb1287 start

