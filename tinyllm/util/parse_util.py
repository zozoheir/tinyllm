import json
import re
from typing import List, Union, Dict, Any

import re
from typing import List

def extract_blocks(text: str, language: str= 'json') -> List[Union[Dict[str, Any], str]]:
    pattern = rf'```{language}\s*(.*?)\s*```'
    matches = re.findall(pattern, text.strip(), re.DOTALL)
    extracted_blocks = [json.loads(match) if language == 'json' else match for match in matches]
    return extracted_blocks

def extract_html(text: str, tag='prompt') -> List[str]:
    pattern = fr'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches
