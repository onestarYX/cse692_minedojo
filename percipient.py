from utils import *
import ast
import google.generativeai as genai

class Percipient:
    def __init__(
        self,
        GOOGLE_API_KEY,
        gemini_model_name,
        temperature=0
    ):
        genai.configure(api_key=GOOGLE_API_KEY)

        self.model = genai.GenerativeModel(gemini_model_name)
    
    def check_sub_objective_success(self, sub_objective, last_frame):
        check_system = load_prompt("check_system")
        check_query = load_prompt("check_query").format(
            structure_information=dict_to_prompt(sub_objective)
        )
        
        downsample_rate = 2
        last_frame_downsampled = last_frame[0::downsample_rate,0::downsample_rate]
        last_frame_downsampled = cv2.cvtColor(last_frame_downsampled, cv2.COLOR_BGR2RGB)
        last_frame_downsampled = Image.fromarray(last_frame_downsampled)
                
        messages = [
            check_system,
            check_query,
            last_frame_downsampled
        ]
        
        last_frame_downsampled.save('last_frame_downsampled.png')

        check_info = self.model.generate_content(messages).text
        check_info = check_info[7:-3]
        
        check_dict = ast.literal_eval(check_info)
        
        assert check_dict["success"] in ["true", "false"]
        if "suggestion" not in check_dict:
            check_dict["suggestion"] = ""
            
        log_info(f"Check Result: {check_dict}")
        
        if check_dict["success"]=="true":
            log_info("**********Sub-objective Success!**********")
        else:
            log_info("**********Sub-objective Failure!**********")

        return check_dict
