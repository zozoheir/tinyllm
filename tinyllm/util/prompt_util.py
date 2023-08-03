import random
from typing import List, Dict, Any, Optional
import re

import tiktoken

from tinyllm.util import os_util

OPENAI_MODELS_MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16384,
    "text-davinci-003 (Legacy)": 4097,
    "text-davinci-002 (Legacy)": 4097,
    "code-davinci-002 (Legacy)": 8001
}


def stringify_list(paragraphs: List[str]) -> str:
    """
    Concatenates a list of strings with newline separator.

    :param paragraphs: A list of strings to concatenate.
    :return: A string concatenated with newline separator.
    """
    return "\n".join(paragraphs)


def stringify_key_value(key: str, value: Any) -> str:
    """
    Formats a string based on a key-value pair.

    :param key: The key of the pair.
    :param value: The value of the pair.
    :return: A formatted string.
    """
    return f"- {key}: {value}"


def stringify_dict(header: str,
                   dict: Dict[str, Any],
                   ignore_keys: Optional[List[str]] = None) -> str:
    """
    Formats a dictionary into a string with a specific format.

    :param dict: A dictionary to format.
    :param ignore_keys: A list of keys to ignore. Default is None.
    :return: A formatted string.
    """
    ignore_keys = ignore_keys or []
    all_strings = []
    for key, value in dict.items():

        if key in ignore_keys or value is None or key is None:
            continue

        if key in ['created_at', 'updated_at', 'timestamp']:
            value = str(value).split('+')[0]

        generated_string = stringify_key_value(key, str(value).split('+')[0])
        all_strings.append(generated_string)

    dict_string_representation = stringify_list(all_strings)
    return header + "\n" + dict_string_representation


def stringify_dict_list(
        dict_header: str,
        dicts: List[Dict[str, Any]],
        ignore_keys: Optional[List[str]] = None) -> str:
    """
    Transforms a list of dictionaries to a single formatted string.

    :param text_header: The header of the text.
    :param dicts: A list of dictionaries to transform.
    :param ignore_keys: A list of keys to omit. Default is None.
    :return: A formatted string.
    """
    ignore_keys = ignore_keys or []
    return stringify_list(
        [stringify_dict(dict_header, data_dict, ignore_keys) for data_dict in dicts])


def remove_imports(code: str) -> str:
    lines = code.split('\n')
    lines = [line for line in lines if not line.lstrip().startswith(('import', 'from'))]
    return '\n'.join(lines)


def extract_markdown_python(text: str):
    pattern = r"```python(.*?)```"
    python_codes = re.findall(pattern, text, re.DOTALL)
    return "\n".join(python_codes)


def get_files_content(file_list: list,
                      formats: list):
    code_context = []
    for file_name in file_list:
        if os_util.isDirPath(file_name):
            for file_path in os_util.listDir(file_name, recursive=True, formats=formats):
                try:
                    with open(file_path, 'r') as file:
                        content = file.read()
                        code_context.append(f'\n \nFILE: This is the content of the file {file_name}:\n \n {content}\n')
                        code_context.append(f'\n------------------------\n')
                except FileNotFoundError:
                    print(f'File {file_name} not found in the directory')

        else:
            try:
                with open(file_name, 'r') as file:
                    content = file.read()
                    code_context.append(f'\n \nFILE: This is the content of the file {file_name}:\n \n {content}\n')
                    code_context.append(f'\n------------------------\n')
            except FileNotFoundError:
                print(f'File {file_name} not found in the directory')

    final_prompt = '\n'.join(code_context)
    return final_prompt


def shuffle_with_freeze(input_list, freeze):
    not_frozen_dict = {i: input_list[i] for i in range(len(input_list)) if i not in freeze}
    not_frozen_indices = list(not_frozen_dict.keys())
    random.shuffle(not_frozen_indices)
    shuffled_dict = {i: not_frozen_dict[not_frozen_indices[i]] for i in range(len(not_frozen_indices))}
    output_list = [shuffled_dict.get(i) if i in shuffled_dict else input_list[i] for i in range(len(input_list))]
    return output_list
