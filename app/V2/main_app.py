# Import libraries and model
import pyautogui # type: ignore
import time # type: ignore
import pydirectinput # type: ignore
import keyboard # type: ignore
import winsound # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
from ultralytics import YOLO  # type: ignore


class CharacterDetection():
    
    def __init__(self,model_path:str,characters:list[str],toggle_key:str,stop_key:str):
        print(f"Loading Model...")
        self.script_alive = True
        self.script_active = False
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded successfully!")
        except Exception as e:
            print(f"Error Loading Model {e}")
        self.toggle_key = toggle_key
        self.stop_key = stop_key
        base_path = 'public/characters/'
        character_image_paths = [f"{base_path}{char_file_name}.png" for char_file_name in characters]
        print('\n'.join([f"Getting Character path : {path}," for path in character_image_paths]))
        self.chars_need = len(character_image_paths)
        try:
            self.template_gray = [cv2.cvtColor(cv2.imread(char_path),cv2.COLOR_BGR2GRAY) for char_path in character_image_paths]
            self.register_hotkeys()
        except FileNotFoundError as e:
            print(f"ERROR: Character File path not found {e}")
            exit()
        
    def exit_script(self):
        self.script_active = False
        self.script_alive = False

    def toggle_script(self):
        self.script_active = not self.script_active
        
        if self.script_active:
            print("--- Script ACTIVATED by Hotkey ---")
        else:
            print("--- Script DEACTIVATED by Hotkey ---")
            
    def register_hotkeys(self):
        try:
            keyboard.add_hotkey(self.toggle_key, self.toggle_script)
            keyboard.add_hotkey(self.stop_key, self.exit_script)

            print(
                f"Script loaded. Press {self.toggle_key.upper()} to Start/Stop. "
                f"Press {self.stop_key.upper()} to Exit."
            )

        except Exception as e:
            print(
                f"Failed to register hotkeys. Try running the script as an administrator. Error: {e}"
            )
            exit()
    
    def detecting_character(self):
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self.model(frame, conf=0.8,verbose=False)
        found_character = []
        for r in results:
            slots = r.boxes
            for slot in slots:
                cls_id = int(slot.cls[0])
                if cls_id == 0: 
                    x1, y1, x2, y2 = map(int, slot.xyxy[0])
                    cropped_character = frame[y1:y2, x1:x2]
                    found_character.append(cropped_character)
        print(f"Finished Scan found {len(found_character)} 5-star characters")
        return found_character
    
    def verify_character(self,cropped_5star_img, threshold=0.6):

     
        cropped_gray = cv2.cvtColor(cropped_5star_img, cv2.COLOR_BGR2GRAY)

        for template_gray in self.template_gray:
            print(f"Now checking : {template_gray} with {cropped_gray}")
            h_temp, w_temp = template_gray.shape
            h_crop, w_crop = cropped_gray.shape
            
            if h_temp > h_crop or w_temp > w_crop:
                print("Template picture is too big ")
                return False

            result = cv2.matchTemplate(cropped_gray, template_gray, cv2.TM_CCOEFF_NORMED)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                print(f"Found With : {max_val*100:.2f}%")
                return True
            else:
                print(f"Not Found : {max_val*100:.2f}%")
        return False      
        
    def start_reroll(self):  
        try:  
            while self.script_alive:
                found = False
                if self.script_active:                
                    character_images = self.detecting_character()
                    target_found = 0
                    for c in character_images:
                        print("Before char check")
                        target_char_check = self.verify_character(c)
                        print(f"After char check {target_char_check}")
                        if target_char_check:
                            target_found += 1
                            print(f"Found {target_found} target now")
                            if target_found == self.chars_need:
                                found = True
                                break
                    self.script_active = False
                    break
                if found:
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
    MODEL_PATH = 'model/train/runs/detect/train/weights/best.pt'

    bot = CharacterDetection(MODEL_PATH,["rafina3","alec"],'f9','esc')
    bot.start_reroll()