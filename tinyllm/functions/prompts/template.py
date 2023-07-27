from typing import List, Type, Dict

from tinyllm.functions.function import Function
from tinyllm.functions.validator import Validator


class OpenAIPromptTemplateInitValidator(Validator):
    messages: List


class OpenAIPromptTemplateOutputValidator(Validator):
    messages: List[Dict[str, str]]

