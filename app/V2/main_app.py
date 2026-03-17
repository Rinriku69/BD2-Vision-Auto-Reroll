# Import libraries and model
import pyautogui 
import time 
import pydirectinput # type: ignore
import keyboard
import winsound 
import cv2 
from cv2.typing import MatLike
import numpy as np 
from ultralytics import YOLO 
from typing import cast
from ultralytics.engine.results import Results


class CharacterDetection():
    
    def __init__(self,char_model_path:str,ui_model_path:str,characters:list[str],toggle_key:str,stop_key:str):
        print(f"Loading Model...")
        # Script State
        self.script_alive: bool = True
        self.script_active: bool = False
        # Load Model
        try:
            self.char_model: YOLO = YOLO(char_model_path)
            self.ui_model: YOLO = YOLO(ui_model_path)
            print(f"Model loaded successfully!")
        except Exception as e:
            print(f"Error Loading Model {e}")
        
        # Register hotkeys
        self.register_hotkeys(toggle_key,stop_key)
        
        # Get target characters images
        base_path = 'public/characters/'
        character_image_paths = [f"{base_path}{char_file_name}.png" for char_file_name in characters]
        print('\n'.join([f"Getting Character path : {path}," for path in character_image_paths]))
        
        # Convert Images to BGR2GRAY
        self.chars_need = len(character_image_paths)
        
        self.template_gray: list[MatLike] = []
        for char_path in character_image_paths:
            img = cv2.imread(char_path)
    
            if img is None:
                raise ValueError(f"โหลดภาพไม่ได้: {char_path}")
    
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.template_gray.append(gray)
     
    # Script State Control    
    def exit_script(self) -> None :
        self.script_active = False
        self.script_alive = False

    def toggle_script(self)  -> None :
        self.script_active = not self.script_active
        
        if self.script_active:
            print("--- Script ACTIVATED by Hotkey ---")
        else:
            print("--- Script DEACTIVATED by Hotkey ---")
            
    def register_hotkeys(self,toggle_key:str,stop_key:str)  -> None :
        try:
            keyboard.add_hotkey(toggle_key, self.toggle_script)
            keyboard.add_hotkey(stop_key, self.exit_script)

            print(
                f"Script loaded. Press {toggle_key.upper()} to Start/Stop. "
                f"Press {stop_key.upper()} to Exit."
            )

        except Exception as e:
            print(
                f"Failed to register hotkeys. Try running the script as an administrator. Error: {e}"
            )
            exit()
    #--------------------
    
       
    #  Screen detection
    def detecting_character(self) -> list[np.ndarray] | None:
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results: list[Results]  = cast(list[Results],self.char_model(frame, conf=0.8,verbose=False))
        found_character: list[np.ndarray] = []
        for r in results:
            slots = r.boxes
            if slots == None:
                return None    
            for slot in slots:
                cls_id = int(slot.cls[0])
                if cls_id == 0: 
                    x1, y1, x2, y2 = map(int, slot.xyxy[0])
                    cropped_character: np.ndarray = frame[y1:y2, x1:x2]
                    found_character.append(cropped_character)
        print(f"Finished Scan found {len(found_character)} 5-star characters")
        return found_character
    
    """ def detecting_ui(self):
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results:list[Results]  = self.char_model(frame, conf=0.8,verbose=False)
        for r in results:
            buttons = r.boxes
            #draw_again = 0, skip = 1
            for btn in buttons:
                cls_btn = int(btn.cls[0])
                if cls_btn == 0:
                    x1, y1, x2, y2 = map(int,btn.xyxy) """
            
    
    def verify_character(self,cropped_5star_img: np.ndarray, threshold:float = 0.6) -> bool:

     
        cropped_gray = cv2.cvtColor(cropped_5star_img, cv2.COLOR_BGR2GRAY)

        for template_gray in self.template_gray:
            print(f"Now checking : {template_gray} with {cropped_gray}")
            h_temp, w_temp = template_gray.shape
            h_crop, w_crop = cropped_gray.shape
            
            if h_temp > h_crop or w_temp > w_crop:
                print("Template picture is too big ")
                return False

            result = cv2.matchTemplate(cropped_gray, template_gray, cv2.TM_CCOEFF_NORMED)

            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                print(f"Found With : {max_val*100:.2f}%")
                return True
            else:
                print(f"Not Found : {max_val*100:.2f}%")
        return False      
        
    def start_reroll(self):  
        try:  
            while self.script_alive:
                condition_met: bool = False
                if self.script_active:                
                    character_images:list[np.ndarray] | None = self.detecting_character()
                    if character_images == None:
                        return #do some thing
                    
                    target_found: int = 0
                    for c in character_images:
                        print("Before char check")
                        target_char_check = self.verify_character(c)
                        print(f"After char check {target_char_check}")
                        if target_char_check:
                            target_found += 1
                            print(f"Found {target_found} target now")
                            if target_found == self.chars_need:
                                condition_met = True
                                break
                    self.script_active = False
                    break
                # Stop Condition
                if condition_met:
                    print("*** Target Founded ****")
                    self.script_active = False
                    print("--- Script Stopped ---")
                    sound_file = "public/sound/tuturu.wav"
                    try:
                        winsound.PlaySound(sound_file, winsound.SND_FILENAME)
                    except Exception as e_sound:
                        print(f"Error playing sound: {e_sound},{sound_file}")
                        winsound.Beep(1000, 500)
                time.sleep(0.1)
        except KeyboardInterrupt: 
            print("\n--- Script interrupted by user (Ctrl+C). Exiting. ---")

        finally:
            print("--- Cleaning up hotkeys... ---")
            keyboard.unhook_all_hotkeys() 
            print("--- Script fully exited. ---")
                
                
if __name__ == "__main__":
    CHAR_MODEL_PATH = 'model/train/runs/detect/char/train/weights/best.pt'
    UI_MODEL_PATH = 'model/train/runs/detect/ui/train2/weights/best.pt'

    bot = CharacterDetection(CHAR_MODEL_PATH,UI_MODEL_PATH,["rafina3","alec"],'f9','esc')
    bot.start_reroll()