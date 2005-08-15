@echo off
echo This will work only with AVRDUDE V 4.3.0 and newer!
echo Select "BOOTLOADER" from the options menu or toggle power or issue a
echo reset via ISP connector. BF must have power-supply (battery or external) 
echo RS232 connection has to be o.k. (can be tested with "UPLOAD NAME").

pause Press joystick ("enter") on BF and ENTER/Return-key on PC-Keyboard to continue

avrdude -v -p m169 -c butterfly -P com1 -e -U flash:w:main.hex

echo Move joystick "up" now and test if the applications starts. 
echo Depending on the AVRDUDE version you have to reset the device if 
echo "AVR-BUTTERFLY GCC" does not appear on the LCD now. Version below
eche 4.4.0 need "hardware" reset or "power toogle".
pause
