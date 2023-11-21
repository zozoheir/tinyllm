import re

from fuzzywuzzy import fuzz
import random
from typing import List, Dict, Any, Optional
import re

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


def stringify_string_list(paragraphs: List[str],
                          separator="\n") -> str:
    """
    Concatenates a list of strings with newline separator.

    :param paragraphs: A list of strings to concatenate.
    :return: A string concatenated with newline separator.
    """
    return separator.join(paragraphs)


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

        if key in ignore_keys:
            continue

        if value is None:
            value=""

        if key in ['created_at', 'updated_at', 'timestamp']:
            value = str(value).split('+')[0]

        generated_string = stringify_key_value(key, str(value).split('+')[0])
        all_strings.append(generated_string)

    dict_string_representation = stringify_string_list(all_strings,
                                                       separator="\n")
    return header + "\n" + dict_string_representation


def stringify_dict_list(
        header: str,
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
    return stringify_string_list(
        paragraphs=[stringify_dict(header, data_dict, ignore_keys) for data_dict in dicts],
        separator="\n\n-------\n\n"
    )


def remove_imports(code: str) -> str:
    lines = code.split('\n')
    lines = [line for line in lines if not line.lstrip().startswith(('import', 'from'))]
    return '\n'.join(lines)


def extract_markdown_python(text: str):
    if '```python' not in text:
        return text
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


def remove_duplicate_lines(input_string: str) -> str:
    lines = input_string.split('\n')
    seen_lines = set()
    unique_lines = []
    for line in lines:
        trimmed_line = line.strip()  # Removing leading and trailing whitespaces
        if trimmed_line and trimmed_line not in seen_lines:
            seen_lines.add(trimmed_line)
            unique_lines.append(trimmed_line)
    return '\n'.join(unique_lines)


def find_closest_match_char_by_char(source, target):
    max_ratio = 0
    best_match = (0, 0)
    n = len(source)

    for start in range(n):
        for end in range(start, n):
            substring = source[start:end + 1]
            ratio = fuzz.token_set_ratio(substring, target)
            if ratio > max_ratio:
                max_ratio = ratio
                best_match = (start, end)

    return best_match

def get_smallest_chunk(source, matches):
    # Sort matches by start index
    matches.sort(key=lambda x: x[0])

    min_chunk = (0, len(source))
    for i in range(len(matches)):
        for j in range(i+1, len(matches)):
            if matches[j][0] > matches[i][1]:  # Ensuring the second element starts after the first
                chunk_size = matches[j][1] - matches[i][0]
                if chunk_size < (min_chunk[1] - min_chunk[0]):
                    min_chunk = (matches[i][0], matches[j][1])
                    break  # No need to check further as we are looking for smallest chunk

    return min_chunk


def preprocess_text(text):
    # Convert to lower case and remove special characters
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())


def split_relationship(input_text):
    # Split the relationship into entities and relationships
    elements = re.findall(r'x:([^\s]+)', input_text)
    return elements


def get_optimal_source_chunk(triplet, source):
    elements = triplet.split(' ')
    source = preprocess_text(source)
    entity_start_end = [
        find_closest_match_char_by_char(source, element) for element in [elements[0], elements[2]]
    ]
    relationship_start_end = find_closest_match_char_by_char(source, elements[1])
    start = min(entity_start_end[0][0], relationship_start_end[0])
    end = max(entity_start_end[1][1], relationship_start_end[1])
    start = max(0, start - 50)
    end = min(len(source), end + 50)
    return start, end

# Test the function with a small example
#relationship = "x:Bitcoin x:has_indicator x:Terminal_Price"
#test_source_text = "Bitcoin's performance is measured by various indicators, including the Terminal Price, which reflects market trends."

# Run the function
#optimal_chunk_start, optimal_chunk_end = get_optimal_source_chunk(relationship, test_source_text)

#print(test_source_text[optimal_chunk_start:optimal_chunk_end])
