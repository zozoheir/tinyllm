import abc
import logging
from collections import OrderedDict
from typing import List

from tinyllm.exceptions import PromptSectionValidationException, UserInputValidationException, \
    InvalidPromptInputException

logger = logging.getLogger(__name__)


class PromptComponent(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self.value = None

    @abc.abstractmethod
    def validate_input(self, value: str) -> bool:
        pass

    def set_value(self, value: str):
        if not self.validate(value):
            raise PromptSectionValidationException(
                "Value is not valid for this PromptComponent"
            )
        self.value = value


class UserInput(PromptComponent):
    def __init__(self,
                 name: str):
        super().__init__(name)

    def validate_input(self,
                 value) -> Exception:
        if not isinstance(value, str):
            raise UserInputValidationException(self.name)


class Section(PromptComponent):
    def __init__(self, name: str, value: str):
        super().__init__(name)
        self.value = value

    def validate_input(self, value: str) -> Exception:
        pass


class Prompt:
    def __init__(self,
                 sections: OrderedDict = None,
                 separator: str = "\n",
                 capitalize_words: List[str] = None,
                 capitalize_section_names: bool = False):
        self.components = sections or OrderedDict()
        self.separator = separator
        self.capitalize_words = capitalize_words or []
        self.capitalize_section_names = capitalize_section_names
        self.formatted_prompt = None

    def get(self):
        formatted_string_parts = []
        for prompt_element in self.components.values():
            section_name = prompt_element.name
            section_value = prompt_element.value
            if self.capitalize_section_names:
                section_name = section_name.upper()
            for word in self.capitalize_words:
                section_value = section_value.replace(word, word.upper())
            formatted_string_parts.append(f"{section_name}\n{section_value}")
        return self.separator.join(formatted_string_parts)

    def validate_inputs(self,
                        **kwargs) -> Exception:
        for key, value in kwargs.items():
            if key not in self.components:
                raise InvalidPromptInputException(f"Invalid key: {key}")
            else:
                self.components[key].validate_input(value)
