import logging
from collections import OrderedDict

from tinyllm import Objects
from tinyllm.exceptions import PromptComponentValidationException, UserInputValidationException, \
    InvalidPromptInputException, UnknownPromptInputException
from tinyllm.function import Function

logger = logging.getLogger(__name__)


class PromptComponent(Function):
    def __init__(self, name: str, input):
        super().__init__(name,
                         input=input,
                         type=Objects.PROMPT_COMPONENT)

    def validate_input(self, **kwargs) -> Exception:
        if not isinstance(value, str):
            raise PromptComponentValidationException(
                "Value is not valid for this PromptComponent"
            )


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
                 components: OrderedDict = None,
                 separator: str = "\n"):
        self.components = components or OrderedDict()
        self.separator = separator
        self.formatted_prompt = None

    def get(self,
            **kwargs) -> str:
        for key, value in kwargs.items():
            if key not in self.components:
                raise UnknownPromptInputException(f"Unknown key: {key}")
            else:
                component = self.components[key]
                component.set_value(value)
        formatted_string_parts = [
            f"{prompt_element.name}\n{prompt_element.value}" for prompt_element in self.components.values()
        ]
        return self.separator.join(formatted_string_parts)

    def validate_inputs(self,
                        **kwargs) -> Exception:
        for key, value in kwargs.items():
            if key not in self.components:
                raise InvalidPromptInputException(f"Invalid key: {key}")
            else:
                self.components[key].validate_input(**kwargs)
