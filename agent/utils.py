import json
import re
from pathlib import Path

def natural_sort_key(s):
    """Sort strings with numbers in a natural way"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_json_files(input_json_dir):
    json_files = [str(json_file) for json_file in Path(input_json_dir).rglob("*.json")]
    for json_file in sorted(json_files, key=natural_sort_key):
        with open(json_file, "r", encoding="utf-8") as f:
            yield json.load(f)

def parse_instructions(input_json_dir_as_instruction):
    instructions = load_json_files(input_json_dir_as_instruction)
    while True:
        try:
            instruction = next(instructions)
            print(instruction)
            print("-" * 100)
        except StopIteration:
            break


    return instructions
