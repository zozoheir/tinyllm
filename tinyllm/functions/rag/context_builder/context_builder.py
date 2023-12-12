from typing import List, Dict, Optional, Callable

from tinyllm.function import Function
from tinyllm.validator import Validator


class ContextBuilderInitValidator(Validator):
    start_string: str
    end_string: str
    available_token_size: int


class ContextBuilderInputValidator(Validator):
    context: str


class ContextBuilderOutputValidator(Validator):
    context: str


class ContextBuilder(Function):
    def __init__(self,
                 start_string: str,
                 end_string: str,
                 available_token_size: int,
                 **kwargs):
        val = ContextBuilderInitValidator(start_string=start_string,
                                          end_string=end_string,
                                          available_token_size=available_token_size)

        super().__init__(
            input_validator=ContextBuilderInputValidator,
            output_validator=ContextBuilderOutputValidator,
            **kwargs
        )
        self.start_string = start_string
        self.end_string = end_string
        self.available_token_size = available_token_size
        self.context = None

    def format_context(self, context: str) -> str:
        final_context = self.start_string + "\n" + context + "\n" + self.end_string
        return final_context

    def run(self, context: str) -> str:
        final_context = \
            self.start_string + "\n" + \
            context + "\n" + \
            self.end_string
        return {'context': final_context}
