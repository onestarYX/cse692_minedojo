from utils import *
import ast
import google.generativeai as genai

class Planner:
    def __init__(
        self,
        GOOGLE_API_KEY,
        gemini_model_name,
        temperature=0
    ):
        genai.configure(api_key=GOOGLE_API_KEY)

        self.model = genai.GenerativeModel(gemini_model_name)
    
    def get_plan(self, sub_objective, check_result):
        structured_action_system = load_prompt("structured_action_system")
        sub_objective_string = dict_to_prompt(sub_objective)

        structured_action_query = load_prompt("structured_action_query").format(
            structure_information=sub_objective_string
        )


        if len(check_result) == 0:
            structured_action_query += "\nPlan your plan. Remember to follow the response format."
        else:
            structured_action_query += f"""The previous plan failed. 
            The reason for the failure: {check_result["feedback"]}.
            A suggested recommendations: {check_result["suggestion"]}. 
            re-plan your workflow. Remember to follow the response format."""
                
        messages = [
            structured_action_system,
            structured_action_query
        ]

        plan = self.model.generate_content(messages).text
        plan = plan[7:-3]
        log_info(f"Create Plan Result: {plan}")

        return ast.literal_eval(plan)
        #return fix_and_parse_json(plan)
