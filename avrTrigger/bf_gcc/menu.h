// menu.h

// mt __flash typedef struct
typedef struct PROGMEM
{
    unsigned char state;
    unsigned char input;
    unsigned char nextstate;
} MENU_NEXTSTATE;


// mt __flash typedef struct
typedef struct PROGMEM
{
    unsigned char state;
    // char __flash *pText;
	PGM_P pText;
    char (*pFunc)(char input);
} MENU_STATE;


// Menu text
// mtA, these where all of the same structure as in the follow. line
// __flash char MT_AVRBF[]                         = "AVR Butterfly";
//const char MT_AVRBF[] PROGMEM                     = "AVR Butterfly GCC";
const char MT_AVRBF[] PROGMEM                     = 
	{'A','V','R',' ','B','U','T','T','E','R','F','L','Y',' ','G'+0x80,'C'+0x80,'C'+0x80,'\0'};
const char MT_TIME[] PROGMEM                      = "Time";
const char MT_TIME_CLOCK[] PROGMEM                = "Clock";
const char MT_TIME_CLOCK_ADJUST[] PROGMEM         = "Adjust Clock";
const char MT_TIME_CLOCKFORMAT_ADJUST[] PROGMEM   = "Change Clock Format";
const char MT_TIME_DATE[] PROGMEM                 = "Date";
const char MT_TIME_DATE_ADJUST[] PROGMEM          = "Adjust Date";
const char MT_TIME_DATEFORMAT_ADJUST[] PROGMEM    = "Change Date Format";
const char MT_MUSIC[] PROGMEM                     = "Music";
const char MT_VCARD[] PROGMEM                     = "Name";
const char MT_ENTERNAME[] PROGMEM                 = "Enter name";
const char MT_TEMPERATURE[] PROGMEM               = "Temperature";
const char MT_VOLTAGE[] PROGMEM                   = "Voltage";
const char MT_LIGHT[] PROGMEM                     = "Light";
const char MT_OPTIONS[] PROGMEM                   = "Options";
const char MT_OPTIONS_DISPLAY[] PROGMEM           = "Display";
const char MT_OPTIONS_DISPLAY_CONTRAST[] PROGMEM  = "Adjust contrast";
// mt const char MT_OPTIONS_DISPLAY_SEG[] PROGMEM       = "Browse segments";
const char MT_VCARD_DOWNLOAD_NAME[] PROGMEM       = "Download Name";
const char MT_OPTIONS_BOOT[] PROGMEM              = "Bootloader";
const char MT_OPTIONS_POWER_SAVE[] PROGMEM        = "Power Save Mode";
const char MT_OPTIONS_AUTO_POWER_SAVE[] PROGMEM   = "Auto Power Save";
const char MT_OPTIONS_KEYCLICK[] PROGMEM          = "Key Click";
// mtE

// mt MENU_NEXTSTATE menu_nextstate[] = { 
const MENU_NEXTSTATE menu_nextstate[] PROGMEM = {
//  STATE                       INPUT       NEXT STATE
    {ST_AVRBF,                  KEY_PLUS,   ST_OPTIONS},
    {ST_AVRBF,                  KEY_NEXT,   ST_AVRBF_REV},
    {ST_AVRBF,                  KEY_MINUS,  ST_TIME},

    {ST_AVRBF_REV,              KEY_PREV,   ST_AVRBF},

    {ST_TIME,                   KEY_PLUS,   ST_AVRBF},
    {ST_TIME,                   KEY_NEXT,   ST_TIME_CLOCK},
    {ST_TIME,                   KEY_PREV,   ST_AVRBF},
    {ST_TIME,                   KEY_MINUS,  ST_MUSIC},

    {ST_TIME_CLOCK,             KEY_PLUS,   ST_TIME_DATE},
    {ST_TIME_CLOCK,             KEY_NEXT,   ST_TIME_CLOCK_FUNC},
    {ST_TIME_CLOCK,             KEY_PREV,   ST_TIME},
    {ST_TIME_CLOCK,             KEY_MINUS,  ST_TIME_DATE},

    {ST_TIME_CLOCK_ADJUST,      KEY_PLUS,   ST_TIME_CLOCKFORMAT_ADJUST}, 
    {ST_TIME_CLOCK_ADJUST,      KEY_ENTER,  ST_TIME_CLOCK_ADJUST_FUNC},
    {ST_TIME_CLOCK_ADJUST,      KEY_PREV,   ST_TIME_CLOCK_FUNC},    
    {ST_TIME_CLOCK_ADJUST,      KEY_MINUS,  ST_TIME_CLOCKFORMAT_ADJUST}, 

    {ST_TIME_CLOCKFORMAT_ADJUST, KEY_PLUS,  ST_TIME_CLOCK_ADJUST},
    {ST_TIME_CLOCKFORMAT_ADJUST, KEY_ENTER, ST_TIME_CLOCKFORMAT_ADJUST_FUNC},
    {ST_TIME_CLOCKFORMAT_ADJUST, KEY_PREV,  ST_TIME_CLOCK_FUNC},
    {ST_TIME_CLOCKFORMAT_ADJUST, KEY_MINUS, ST_TIME_CLOCK_ADJUST},

    {ST_TIME_DATE,              KEY_PLUS,   ST_TIME_CLOCK},
    {ST_TIME_DATE,              KEY_NEXT,   ST_TIME_DATE_FUNC},
    {ST_TIME_DATE,              KEY_PREV,   ST_TIME},
    {ST_TIME_DATE,              KEY_MINUS,  ST_TIME_CLOCK},

    {ST_TIME_DATE_ADJUST,       KEY_PLUS,   ST_TIME_DATEFORMAT_ADJUST},
    {ST_TIME_DATE_ADJUST,       KEY_ENTER,  ST_TIME_DATE_ADJUST_FUNC},
    {ST_TIME_DATE_ADJUST,       KEY_PREV,   ST_TIME_DATE_FUNC},
    {ST_TIME_DATE_ADJUST,       KEY_MINUS,  ST_TIME_DATEFORMAT_ADJUST},

    {ST_TIME_DATEFORMAT_ADJUST, KEY_PLUS,   ST_TIME_DATE_ADJUST},
    {ST_TIME_DATEFORMAT_ADJUST, KEY_ENTER,  ST_TIME_DATEFORMAT_ADJUST_FUNC},
    {ST_TIME_DATEFORMAT_ADJUST, KEY_PREV,   ST_TIME_DATE_FUNC},
    {ST_TIME_DATEFORMAT_ADJUST, KEY_MINUS,  ST_TIME_DATE_ADJUST},

    {ST_MUSIC,                  KEY_PLUS,   ST_TIME},
    {ST_MUSIC,                  KEY_NEXT,   ST_MUSIC_SELECT},
    {ST_MUSIC,                  KEY_PREV,   ST_AVRBF},
    {ST_MUSIC,                  KEY_MINUS,  ST_VCARD},

    {ST_SOUND_MUSIC,            KEY_NEXT,   ST_MUSIC_SELECT},
    {ST_SOUND_MUSIC,            KEY_PREV,   ST_MUSIC},

    {ST_VCARD,                  KEY_PLUS,   ST_MUSIC},
    {ST_VCARD,                  KEY_NEXT,   ST_VCARD_FUNC},
    {ST_VCARD,                  KEY_PREV,   ST_AVRBF},
    {ST_VCARD,                  KEY_MINUS,  ST_TEMPERATURE},

    {ST_VCARD_ENTER_NAME,       KEY_PLUS,     ST_VCARD_DOWNLOAD_NAME},
    {ST_VCARD_ENTER_NAME,       KEY_ENTER,    ST_VCARD_ENTER_NAME_FUNC},
    {ST_VCARD_ENTER_NAME,       KEY_PREV,     ST_VCARD_FUNC},    
    {ST_VCARD_ENTER_NAME,       KEY_MINUS,    ST_VCARD_DOWNLOAD_NAME},

    {ST_VCARD_DOWNLOAD_NAME,    KEY_PLUS,     ST_VCARD_ENTER_NAME},
    {ST_VCARD_DOWNLOAD_NAME,    KEY_ENTER,    ST_VCARD_DOWNLOAD_NAME_FUNC},
    {ST_VCARD_DOWNLOAD_NAME,    KEY_PREV,     ST_VCARD_FUNC},    
    {ST_VCARD_DOWNLOAD_NAME,    KEY_MINUS,    ST_VCARD_ENTER_NAME},    

    {ST_TEMPERATURE,            KEY_PLUS,   ST_VCARD},
    {ST_TEMPERATURE,            KEY_NEXT,   ST_TEMPERATURE_FUNC},
    {ST_TEMPERATURE,            KEY_PREV,   ST_AVRBF},
    {ST_TEMPERATURE,            KEY_MINUS,  ST_VOLTAGE},

    {ST_VOLTAGE,                KEY_PLUS,   ST_TEMPERATURE},
    {ST_VOLTAGE,                KEY_NEXT,   ST_VOLTAGE_FUNC},
    {ST_VOLTAGE,                KEY_PREV,   ST_AVRBF},
    {ST_VOLTAGE,                KEY_MINUS,  ST_LIGHT},

    {ST_LIGHT,                  KEY_PLUS,   ST_VOLTAGE},
    {ST_LIGHT,                  KEY_NEXT,   ST_LIGHT_FUNC},
    {ST_LIGHT,                  KEY_PREV,   ST_AVRBF},
    {ST_LIGHT,                  KEY_MINUS,  ST_OPTIONS},

    {ST_OPTIONS,                KEY_PLUS,   ST_LIGHT},
    {ST_OPTIONS,                KEY_NEXT,   ST_OPTIONS_DISPLAY},
    {ST_OPTIONS,                KEY_PREV,   ST_AVRBF},
    {ST_OPTIONS,                KEY_MINUS,  ST_AVRBF},

    {ST_OPTIONS_DISPLAY,        KEY_PLUS,   ST_OPTIONS_KEYCLICK},
    {ST_OPTIONS_DISPLAY,        KEY_NEXT,   ST_OPTIONS_DISPLAY_CONTRAST},
    {ST_OPTIONS_DISPLAY,        KEY_PREV,   ST_OPTIONS},
    {ST_OPTIONS_DISPLAY,        KEY_MINUS,  ST_OPTIONS_BOOT},

    {ST_OPTIONS_DISPLAY_CONTRAST, KEY_ENTER,    ST_OPTIONS_DISPLAY_CONTRAST_FUNC},
    {ST_OPTIONS_DISPLAY_CONTRAST, KEY_PREV,     ST_OPTIONS_DISPLAY},

    {ST_OPTIONS_BOOT,             KEY_PLUS,     ST_OPTIONS_DISPLAY},
    {ST_OPTIONS_BOOT,             KEY_NEXT,     ST_OPTIONS_BOOT_FUNC},
    {ST_OPTIONS_BOOT,             KEY_PREV,     ST_OPTIONS},
    {ST_OPTIONS_BOOT,             KEY_MINUS,    ST_OPTIONS_POWER_SAVE},

    {ST_OPTIONS_POWER_SAVE,       KEY_PLUS,     ST_OPTIONS_BOOT},
    {ST_OPTIONS_POWER_SAVE,       KEY_NEXT,     ST_OPTIONS_POWER_SAVE_FUNC},
    {ST_OPTIONS_POWER_SAVE,       KEY_PREV,     ST_OPTIONS},
    {ST_OPTIONS_POWER_SAVE,       KEY_MINUS,    ST_OPTIONS_AUTO_POWER_SAVE},

    {ST_OPTIONS_AUTO_POWER_SAVE,  KEY_PLUS,     ST_OPTIONS_POWER_SAVE},
    {ST_OPTIONS_AUTO_POWER_SAVE,  KEY_NEXT,     ST_OPTIONS_AUTO_POWER_SAVE_FUNC},
    {ST_OPTIONS_AUTO_POWER_SAVE,  KEY_PREV,     ST_OPTIONS},
    {ST_OPTIONS_AUTO_POWER_SAVE,  KEY_MINUS,    ST_OPTIONS_KEYCLICK},

    {ST_OPTIONS_KEYCLICK,         KEY_PLUS,     ST_OPTIONS_AUTO_POWER_SAVE},
    {ST_OPTIONS_KEYCLICK,         KEY_NEXT,     ST_OPTIONS_KEYCLICK_FUNC},
    {ST_OPTIONS_KEYCLICK,         KEY_PREV,     ST_OPTIONS},
    {ST_OPTIONS_KEYCLICK,         KEY_MINUS,    ST_OPTIONS_DISPLAY},
        
    {0,                         0,          0}
};


// mt MENU_STATE menu_state[] = {
const MENU_STATE menu_state[] PROGMEM = {
//  STATE                               STATE TEXT                  STATE_FUNC
    {ST_AVRBF,                          MT_AVRBF,                   NULL},
    {ST_AVRBF_REV,                      NULL,                       Revision},
    {ST_TIME,                           MT_TIME,                    NULL},
    {ST_TIME_CLOCK,                     MT_TIME_CLOCK,              NULL},
    {ST_TIME_CLOCK_FUNC,                NULL,                       ShowClock},
    {ST_TIME_CLOCK_ADJUST,              MT_TIME_CLOCK_ADJUST,       NULL},
    {ST_TIME_CLOCK_ADJUST_FUNC,         NULL,                       SetClock},
    {ST_TIME_CLOCKFORMAT_ADJUST,        MT_TIME_CLOCKFORMAT_ADJUST, NULL},
    {ST_TIME_CLOCKFORMAT_ADJUST_FUNC,   NULL,                       SetClockFormat},
    {ST_TIME_DATE,                      MT_TIME_DATE,               NULL},
    {ST_TIME_DATE_FUNC,                 NULL,                       ShowDate},
    {ST_TIME_DATE_ADJUST,               MT_TIME_DATE_ADJUST,        NULL},
    {ST_TIME_DATE_ADJUST_FUNC,          NULL,                       SetDate},    
    {ST_TIME_DATEFORMAT_ADJUST,         MT_TIME_DATEFORMAT_ADJUST,  NULL},
    {ST_TIME_DATEFORMAT_ADJUST_FUNC,    NULL,                       SetDateFormat},     
    {ST_MUSIC,                          MT_MUSIC,                   NULL},
    {ST_MUSIC_SELECT,                   NULL,                       SelectSound},
    {ST_MUSIC_PLAY,                     NULL,                       Sound},
    {ST_VCARD,                          MT_VCARD,                   NULL},
    {ST_VCARD_FUNC,                     NULL,                       vCard},
    {ST_VCARD_ENTER_NAME,               MT_ENTERNAME,               NULL},
    {ST_VCARD_DOWNLOAD_NAME,            MT_VCARD_DOWNLOAD_NAME,     NULL},
    {ST_VCARD_ENTER_NAME_FUNC,          NULL,                       EnterName},
    {ST_VCARD_DOWNLOAD_NAME_FUNC,       NULL,                       RS232},
    {ST_TEMPERATURE,                    MT_TEMPERATURE,             NULL},
    {ST_TEMPERATURE_FUNC,               NULL,                       TemperatureFunc},
    {ST_VOLTAGE,                        MT_VOLTAGE,                 NULL},
    {ST_VOLTAGE_FUNC,                   NULL,                       VoltageFunc},
    {ST_LIGHT,                          MT_LIGHT,                   NULL},
    {ST_LIGHT_FUNC,                     NULL,                       LightFunc},
    {ST_OPTIONS,                        MT_OPTIONS,                 NULL},
    {ST_OPTIONS_DISPLAY,                MT_OPTIONS_DISPLAY,         NULL},
    {ST_OPTIONS_DISPLAY_CONTRAST,       MT_OPTIONS_DISPLAY_CONTRAST,NULL},
    {ST_OPTIONS_DISPLAY_CONTRAST_FUNC,  NULL,                       SetContrast},
    {ST_OPTIONS_BOOT,                   MT_OPTIONS_BOOT,            NULL},
    {ST_OPTIONS_BOOT_FUNC,              NULL,                       BootFunc},    
    {ST_OPTIONS_POWER_SAVE,             MT_OPTIONS_POWER_SAVE,      NULL},
    {ST_OPTIONS_POWER_SAVE_FUNC,        NULL,                       PowerSaveFunc},
    {ST_OPTIONS_AUTO_POWER_SAVE,        MT_OPTIONS_AUTO_POWER_SAVE, NULL},
    {ST_OPTIONS_AUTO_POWER_SAVE_FUNC,   NULL,                       AutoPower},
    {ST_OPTIONS_KEYCLICK,               MT_OPTIONS_KEYCLICK,        NULL},
    {ST_OPTIONS_KEYCLICK_FUNC,          NULL,                       KeyClick},

    {0,                                 NULL,                       NULL},
    
};
