# WinAVR Sample makefile (c) 2002-2003 Eric B. Weddington
# Released to the Public Domain
# Please read the make user manual!
#
#
# On command line:
# make all = Make software.
# make clean = Clean out built project files.
# make coff = Convert ELF to COFF using objtool.
#
# To rebuild project do make clean then make all.
#


# MCU name
#MCU = at90s2313
MCU = atmega169

# Output format. (can be srec, ihex)
FORMAT = ihex

# Target file name (without extension).
#TARGET = trigger_main
#TARGET = blink
TARGET = trig_serial

# Optimization level (can be 0, 1, 2, 3, s) 
# (Note: 3 is not always the best optimization level. See avr-libc FAQ)
#OPT = s
OPT = 2

# List C source files here. (C dependencies are automatically generated.)
# LCD_driver.c LCD_functions.c BCD.c
SRC = $(TARGET).c ser169.c

# List Assembler source files here.
ASRC = 

# Optional compiler flags.
CFLAGS = -g -O$(OPT) -funsigned-char -funsigned-bitfields -fpack-struct \
-fshort-enums -Wall -Wstrict-prototypes -Wa,-ahlms=$(<:.c=.lst)

# Optional assembler flags.
ASFLAGS = -Wa,-ahlms=$(<:.s=.lst), -gstabs 

# Optional linker flags.
LDFLAGS = -Wl,-Map=$(TARGET).map,--cref

# Additional library flags (-lm = math library).
LIBFLAGS = -lm



# ---------------------------------------------------------------------------

# Define directories, if needed.
#DIRAVR = c:/winavr
#DIRAVRBIN = $(DIRAVR)/bin
#DIRAVRUTILS = $(DIRAVR)/utils/bin
#DIRINC = .
#DIRLIB = $(DIRAVR)/avr/lib


# Define programs and commands.
SHELL = sh

CC = avr-gcc

OBJCOPY = avr-objcopy
OBJDUMP = avr-objdump

REMOVE = rm -f
COPY = cp

ELFCOFF = objtool

HEXSIZE = @avr-size --target=$(FORMAT) $(TARGET).hex
ELFSIZE = @avr-size $(TARGET).elf

FINISH = @echo Errors: none
BEGIN = @echo -------- begin --------
END = @echo --------  end  --------




# Define all object files.
OBJ = $(SRC:.c=.o) $(ASRC:.s=.o) 

# Define all listing files.
LST = $(ASRC:.s=.lst) $(SRC:.c=.lst)

# Combine all necessary flags and optional flags. Add target processor to flags.
ALL_CFLAGS = -mmcu=$(MCU) -I. $(CFLAGS)
ALL_ASFLAGS = -mmcu=$(MCU) -I. -x assembler-with-cpp $(ASFLAGS)
ALL_LDFLAGS = -mmcu=$(MCU) $(LDFLAGS)



# Default target.
all: begin gccversion sizebefore $(TARGET).elf $(TARGET).hex $(TARGET).eep $(TARGET).lss sizeafter finished end


# Eye candy.
begin:
	$(BEGIN)

finished:
	$(FINISH)

end:
	$(END)


# Display size of file.
sizebefore:
	@echo Size before:
	-$(HEXSIZE)

sizeafter:
	@echo Size after:
	$(HEXSIZE)



# Display compiler version information.
gccversion : 
	$(CC) --version




# Target: Convert ELF to COFF for use in debugging / simulating in AVR Studio.
coff: $(TARGET).cof end

%.cof: %.elf
	$(ELFCOFF) loadelf $< mapfile $*.map writecof $@




# Create final output files (.hex, .eep) from ELF output file.
%.hex: %.elf
	$(OBJCOPY) -O $(FORMAT) -R .eeprom $< $@

%.eep: %.elf
	-$(OBJCOPY) -j .eeprom --set-section-flags=.eeprom="alloc,load" --change-section-lma .eeprom=0 -O $(FORMAT) $< $@

# Create extended listing file from ELF output file.
%.lss: %.elf
	$(OBJDUMP) -h -S $< > $@



# Link: create ELF output file from object files.
.SECONDARY : $(TARGET).elf
.PRECIOUS : $(OBJ)
%.elf: $(OBJ)
	$(CC) $(ALL_LDFLAGS) $(OBJ) $(LIBFLAGS) --output $@


# Compile: create object files from C source files.
%.o : %.c
	$(CC) -c $(ALL_CFLAGS) $< -o $@


# Assemble: create object files from assembler source files.
%.o : %.s
	$(CC) -c $(ALL_ASFLAGS) $< -o $@






# Target: clean project.
clean: begin clean_list finished end

clean_list :
	$(REMOVE) $(TARGET).hex
	$(REMOVE) $(TARGET).eep
	$(REMOVE) $(TARGET).obj
	$(REMOVE) $(TARGET).cof
	$(REMOVE) $(TARGET).elf
	$(REMOVE) $(TARGET).map
	$(REMOVE) $(TARGET).obj
	$(REMOVE) $(TARGET).a90
	$(REMOVE) $(TARGET).sym
	$(REMOVE) $(TARGET).lnk
	$(REMOVE) $(TARGET).lss
	$(REMOVE) $(OBJ)
	$(REMOVE) $(LST)


# Automatically generate C source code dependencies. (Code taken from the GNU make user manual.)
# Note that this will work with sh (bash) and sed that is shipped with WinAVR (see the SHELL variable defined above).
# This may not work with other shells or other seds.
%.d: %.c
	set -e; $(CC) -MM $(ALL_CFLAGS) $< \
	| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
	[ -s $@ ] || rm -f $@


# Remove the '-' if you want to see the dependency files generated.
-include $(SRC:.c=.d)



# Listing of phony targets.
.PHONY : all begin finish end sizebefore sizeafter gccversion coff clean clean_list
