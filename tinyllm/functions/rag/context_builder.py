from typing import List, Dict, Optional, Callable

from tinyllm.function import Function
from tinyllm.validator import Validator

class InitValidator(Validator):
    collection_name: str

class InputValidator(Validator):
    input: str
    k: Optional[int] = 1

class ContextBuilderInputValidator(Validator):
    context: str

class ContextBuilderOutputValidator(Validator):
    context: str

class ContextBuilder(Function):
    def __init__(self,
                 **kwargs):
        super().__init__(
            input_validator=ContextBuilderInputValidator,
            output_validator=ContextBuilderOutputValidator,
            **kwargs
        )

    def format_context(self, context: str) -> str:
        final_context = self.start_string + "\n" + context + "\n" + self.end_string
        return final_context


    def run(self, context: str) -> str:
        final_context = \
            self.start_string + "\n" + \
            context + "\n" + \
            self.end_string
        return {'context': final_context}
