import pyautogui # type: ignore
import time
import pydirectinput # type: ignore
import keyboard # type: ignore
import winsound # type: ignore
import pyscreeze # type: ignore

# Character File Name (INPUT)
BASE_PATH = "public/characters/"
CHAR_FILE_NAME = ["ventana1","liberta1"]
TARGET_CHAR_IMAGE = [f"{BASE_PATH}{CHAR}.png" for CHAR in CHAR_FILE_NAME]


# Position For Star (INPUT)
SLOT_TOP_Y_STARS = 653
SLOT_BOTTOM_Y_STARS = 674 
FIRST_SLOT_LEFT_X_STARS = 370
SLOT_WIDTH_STARS = 95

# Slot Space (INPUT)
SLOT_SPACING = 25 

# Character Image (INPUT)
TARGET_CHAR_SLOT_TOP_Y = 398
TARGET_CHAR_SLOT_BOTTOM_Y = 656 
TARGET_CHAR_FIRST_SLOT_LEFT_X = 371
TARGET_CHAR_SLOT_WIDTH = 98

# Button Position (INPUT)
DRAW_AGAIN_X, DRAW_AGAIN_Y = 1462, 999
SKIP_X, SKIP_Y = 1765, 59

# Hotkey (INPUT)
SCRIPT_TOGGLE_KEY = 'f9'
SCRIPT_QUIT_KEY = 'esc'

# -------SETTING--------------------

# Condition (INPUT)
REQUIRED_5_STAR_COUNT = 1

# Delays
DELAY_AFTER_CLICK = 0.5
DELAY_AFTER_PULL_SEQUENCE = 0.5 
DELAY_BETWEEN_SLOT_CHECKS = 0.02
DELAY_BEFORE_RETRY = 0.2

# Image search confidence (0.0 to 1.0, higher is stricter)
FIVE_STAR_CONFIDENCE = 0.8
TARGET_CHAR_CONFIDENCE = 0.7

# -----------------------------------

#Script State
script_active = False
keep_script_running = True


# 5 star image
FIVE_STAR_IMAGE = "public/fivestar.png"
NUMBER_OF_SLOTS_TO_CHECK = 10



#-------- Script State Functions ---------
def toggle_script_state():
    global script_active
    script_active = not script_active
    if script_active:
        print("--- Script ACTIVATED by Hotkey ---")
    else:
        print("--- Script DEACTIVATED by Hotkey ---")

def quit_script():
    global keep_script_running, script_active
    print("--- EXIT Hotkey pressed. Stopping script... ---")
    script_active = False 
    keep_script_running = False 
#
#--------------------------------------

#Hotkey register
try:
    keyboard.add_hotkey(SCRIPT_TOGGLE_KEY, toggle_script_state) # to start/stop
    keyboard.add_hotkey(SCRIPT_QUIT_KEY, quit_script)      # to quit the script entirely
    print(f"Script loaded. Press {SCRIPT_TOGGLE_KEY.upper()} to Start/Stop. Press {SCRIPT_QUIT_KEY.upper()} to Exit.")
except Exception as e:
    print(f"Failed to register hotkeys. Try running the script as an administrator. Error: {e}")
    print("Exiting script.")
    exit()



# ****** DEBUG ZONE 🛠️ **********

#Debug for buttons position
'''pyautogui.click(DRAW_AGAIN_X, DRAW_AGAIN_Y)
time.sleep(DELAY_AFTER_CLICK)
pyautogui.press('enter')
time.sleep(DELAY_AFTER_CLICK)

for y in range(10): 
    pyautogui.click(SKIP_X, SKIP_Y)
    time.sleep(0.2) '''

#Debug for specific slot
""" slot_to_check = 9
current_slot_x_stars = FIRST_SLOT_LEFT_X_STARS + (slot_to_check * (SLOT_WIDTH_STARS + SLOT_SPACING))
star_search_region = (current_slot_x_stars, SLOT_TOP_Y_STARS, SLOT_WIDTH_STARS, SLOT_BOTTOM_Y_STARS - SLOT_TOP_Y_STARS)
location = pyautogui.locateOnScreen(FIVE_STAR_IMAGE, confidence=FIVE_STAR_CONFIDENCE, region=star_search_region)

if location:
     print(f"Found 5-star at {location}")
     pyautogui.click(pyautogui.center(location)) 
else:
     print("5-star not found in this slot.") """

#Debug for search loop
total_pull = 0
while keep_script_running:
  if script_active: 

    target_character_found_this_pull = False
    found_5_star_count = 0

    # ------[][][][][] Loop for Search on each slot [][][][][]---------

    for i in range(NUMBER_OF_SLOTS_TO_CHECK):
     if not script_active: break 
     current_slot_num = i + 1
     
     #==========  5  STAR SEARCH =========================
     
     #Find Star Area Position
        # --Find most left position of star
     current_slot_x_stars = FIRST_SLOT_LEFT_X_STARS + (i * (SLOT_WIDTH_STARS + SLOT_SPACING))
        # --Find star area
     star_search_region = (current_slot_x_stars, SLOT_TOP_Y_STARS, SLOT_WIDTH_STARS, SLOT_BOTTOM_Y_STARS - SLOT_TOP_Y_STARS)
     #--------------------------------

    # Check if 5 star found in current slot
     try: 
        if pyautogui.locateOnScreen(FIVE_STAR_IMAGE, confidence=FIVE_STAR_CONFIDENCE, region=star_search_region):
         found_5_star_count += 1
         print(f"5 Star Found at SLOT #{current_slot_num}")
     except Exception as e: 
        pass 
    #-------------------------------------
    
    #====================================================
    
    
    #============ TARGET CHARACTER SEARCH =====================
    
    #Find Target Character Image Area Position
        # --Find most left on X Axis
     current_slot_x_target = FIRST_SLOT_LEFT_X_STARS + (i * (SLOT_WIDTH_STARS + SLOT_SPACING))
        # --Find image position
     target_search_region = (current_slot_x_target, TARGET_CHAR_SLOT_TOP_Y, TARGET_CHAR_SLOT_WIDTH, TARGET_CHAR_SLOT_BOTTOM_Y - TARGET_CHAR_SLOT_TOP_Y)
    #--------------------------------
    
    # Check if all target characters already found 
     if not target_character_found_this_pull:
        # Check if image pixel match with the target
      target_found = []
      for char_file_name in TARGET_CHAR_IMAGE:
        if char_file_name in target_found:
            continue
        else:
            try:
                if pyautogui.locateOnScreen(char_file_name, confidence=TARGET_CHAR_CONFIDENCE, region=target_search_region):
                    print(f"Found {char_file_name} at Slot #{current_slot_num}")
                    target_found.append(char_file_name)
                    if len(target_found) >= len(TARGET_CHAR_IMAGE):
                        target_character_found_this_pull = True
            except Exception as e:
                    print(f"{char_file_name} not found At slot #{current_slot_num}")


    #-----STOP CONDITION-------
    if (found_5_star_count>=REQUIRED_5_STAR_COUNT and target_character_found_this_pull) :
         total_pull = total_pull + 10
         print(f"Condition met with {total_pull} pulls! Stopping script. Press Esc to Exit.")
         script_active = False 
         sound_file = "public/sound/tuturu.wav"
         try:
          winsound.PlaySound(sound_file, winsound.SND_FILENAME)
         except Exception as e_sound:
          print(f"Error playing sound: {e_sound},{sound_file}")
          winsound.Beep(1000, 500)
    else:
         total_pull = total_pull + 10
         print(f"Conditions not met, re-pulling after a delay... current total pulls {total_pull}")
         time.sleep(DELAY_BEFORE_RETRY)
    script_active = False 
    print("SCRIPT END")
  else:
    time.sleep(0.1) 
   #----------------------------
#
# **********************************