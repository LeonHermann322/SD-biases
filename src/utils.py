import json
from typing import Callable

def for_each_prompt(prompts_file_path: str, folder: str, func: Callable[[str, str, str, list[str]], None]) -> None:
    with open(prompts_file_path, "r") as f:
        prompts = json.load(f)

    for key in prompts.keys():
        prompt_dict = prompts[key]
        prefixes = prompt_dict["prefix"]
        objects = prompt_dict["object"]

        for obj in objects:
            func(folder, key, obj, prefixes)