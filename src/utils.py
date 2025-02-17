import json
from typing import Callable

def for_each_prompt(prompts_file_path: str, folder: str, setting: str, func: Callable[[str, str, str, str, list[str], int], None]) -> None:
    with open(prompts_file_path, "r") as f:
        prompts = json.load(f)

    for key in prompts.keys():
        prompt_dict = prompts[key]
        prefixes = prompt_dict["prefix"]
        objects = prompt_dict["object"]
        images_per_prompt = prompt_dict["images_per_prompt"]

        for obj in objects:
            func(folder, setting, key, obj, prefixes, images_per_prompt)