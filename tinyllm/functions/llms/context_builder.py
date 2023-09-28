from typing import List, Dict, Any, Optional
from tinyllm.functions.function import Function
from tinyllm.functions.llms.open_ai.util.helpers import count_tokens, get_user_message
from tinyllm.functions.validator import Validator
from abc import ABC, abstractmethod

from tinyllm.util.prompt_util import stringify_dict_list


class ContextBuilderInitValidator(Validator):
    start_string: str
    end_string: str
    available_token_size: int

class ContextBuilderInputValidator(Validator):
    docs: Any
    output_format: str = "str"
    header: Optional[str] = None
    ignore_keys: Optional[List[str]] = []

class ContextBuilderOutputValidator(Validator):
    context: str

class ContextBuilder(Function, ABC):
    def __init__(self,
                 start_string: str,
                 end_string: str,
                 available_token_size: int,
                 **kwargs):
        val = ContextBuilderInitValidator(
            start_string=start_string,
            end_string=end_string,
            available_token_size=available_token_size
        )
        super().__init__(
            input_validator=ContextBuilderInputValidator,
            output_validator=ContextBuilderOutputValidator,
            **kwargs
        )
        self.start_string = start_string
        self.end_string = end_string
        self.available_token_size = available_token_size
        self.input = None
        self.fitted_input = None
        self.context = None

    async def run(self, **kwargs):
        return {"context": self.get_context(**kwargs)}

    def get_context(self, docs, output_format="str", **kwargs):
        self.input = docs

        # If list of strings, convert to list of dicts
        if isinstance(docs, List) and isinstance(docs[0], str):
            docs = [{"content": string} for string in docs]

        # If list of dictionaries
        if isinstance(docs, List) and isinstance(docs[0], Dict):
            fitted_list = self.fit(docs,
                                   **kwargs)
            fitted_context_string = stringify_dict_list(header=kwargs.get("header", "[doc]"),
                                                        dicts=fitted_list,
                                                        ignore_keys=kwargs.get("ignore_keys", []))

        formatted_context = self.format_context(fitted_context_string)
        if output_format == "openai":
            return get_user_message(formatted_context)
        elif output_format == "str":
            return formatted_context

    def format_context(self,
                       context: str):
        final_context = self.start_string + "\n" + context + "\n" + self.end_string
        return final_context

    @abstractmethod
    def fit(self, input, **kwargs):
        pass


class DocsContextBuilder(ContextBuilder):
    def fit(self, docs: List[Dict], **kwargs):
        fitted_list = []
        for doc in docs:
            if count_tokens(fitted_list, **kwargs) + count_tokens(doc, **kwargs) < self.available_token_size:
                fitted_list.append(doc)
            else:
                break
        return fitted_list
