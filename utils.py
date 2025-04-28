from typing import Any, Dict, Union
import numpy as np
import logging
import json
import cv2
from PIL import Image


def place_block(env: Any, block_type: str, pos: np.ndarray):
    # Get the agent's position
    null_act = env.action_space.no_op()
    obs, reward, done, info = env.step(null_act)
    agent_pos = obs['location_stats']['pos']

    rel_pos = pos - agent_pos
    env.set_block(block_type, rel_pos)
    

def render_rgb(env):
    null_act = env.action_space.no_op()
    obs, reward, done, info = env.step(null_act)
    frame = np.transpose(obs['rgb'], (1, 2, 0))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame
    
    
def extract_char_position(error_message: str) -> int:
    """Extract the character position from the JSONDecodeError message.
    Args:
        error_message (str): The error message from the JSONDecodeError
          exception.
    Returns:
        int: The character position.
    """
    import re

    char_pattern = re.compile(r"\(char (\d+)\)")
    if match := char_pattern.search(error_message):
        return int(match[1])
    else:
        raise ValueError("Character position not found in the error message.")


def add_quotes_to_property_names(json_string: str) -> str:
    """
    Add quotes to property names in a JSON string.
    Args:
        json_string (str): The JSON string.
    Returns:
        str: The JSON string with quotes added to property names.
    """

    def replace_func(match):
        return f'"{match.group(1)}":'

    property_name_pattern = re.compile(r"(\w+):")
    corrected_json_string = property_name_pattern.sub(replace_func, json_string)

    try:
        json.loads(corrected_json_string)
        return corrected_json_string
    except json.JSONDecodeError as e:
        raise e


def balance_braces(json_string: str) -> str:
    """
    Balance the braces in a JSON string.
    Args:
        json_string (str): The JSON string.
    Returns:
        str: The JSON string with braces balanced.
    """

    open_braces_count = json_string.count("{")
    close_braces_count = json_string.count("}")

    while open_braces_count > close_braces_count:
        json_string += "}"
        close_braces_count += 1

    while close_braces_count > open_braces_count:
        json_string = json_string.rstrip("}")
        close_braces_count -= 1

    try:
        json.loads(json_string)
        return json_string
    except json.JSONDecodeError as e:
        raise e


def fix_invalid_escape(json_str: str, error_message: str) -> str:
    while error_message.startswith("Invalid \\escape"):
        bad_escape_location = extract_char_position(error_message)
        json_str = json_str[:bad_escape_location] + json_str[bad_escape_location + 1 :]
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            error_message = str(e)
    return json_str
    
    
def correct_json(json_str: str) -> str:
    """
    Correct common JSON errors.
    Args:
        json_str (str): The JSON string.
    """

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        error_message = str(e)
        if error_message.startswith("Invalid \\escape"):
            json_str = fix_invalid_escape(json_str, error_message)
        if error_message.startswith(
            "Expecting property name enclosed in double quotes"
        ):
            json_str = add_quotes_to_property_names(json_str)
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError as e:
                error_message = str(e)
        if balanced_str := balance_braces(json_str):
            return balanced_str
    return json_str
    
 
def fix_and_parse_json(
    json_str: str, try_to_fix_with_gpt: bool = True
) -> Union[str, Dict[Any, Any]]:
    """Fix and parse JSON string"""
    try:
        json_str = json_str.replace("\t", "")
        return json.loads(json_str)
    except json.JSONDecodeError as _:  # noqa: F841
        json_str = correct_json(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as _:  # noqa: F841
            pass
    try:
        brace_index = json_str.index("{")
        json_str = json_str[brace_index:]
        last_brace_index = json_str.rindex("}")
        json_str = json_str[: last_brace_index + 1]
        return json.loads(json_str)
    except json.JSONDecodeError as e:  # noqa: F841
        raise e
        
        
def log_info(info, is_logging=True):
    if not is_logging:
        logging.disable(logging.CRITICAL + 1)

    logging.info(info)

    if not is_logging:
        logging.disable(logging.NOTSET)

    print(info)
    
    
def load_text(fpaths, by_lines=False):
    with open(fpaths, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()

def load_prompt(prompt):
    return load_text(f"prompts/{prompt}.txt")
    


def dict_to_prompt(dict_data):
    text = ""
    for key in dict_data:
        if dict_data[key]:
            text += f"- {key}: {dict_data[key]}\n"
        else:
            text += f"- {key}: None\n"
    return text
