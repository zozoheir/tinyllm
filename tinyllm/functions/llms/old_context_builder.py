from abc import abstractmethod
from typing import List, Dict, Any, Optional

from tinyllm.functions.llms.open_ai.util.helpers import get_user_message, count_tokens
from tinyllm.util.prompt_util import stringify_dict_list, stringify_string_list


class ContextBuilder:
    def __init__(self,
                 start_string: str,
                 end_string: str,
                 available_token_size: int):
        self.start_string = start_string
        self.end_string = end_string
        self.available_token_size = available_token_size
        self.input = None
        self.fitted_input = None
        self.context = None

    def get_context(self,
                    docs: Any,
                    output_format: str = "str",
                    **kwargs):
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
    def fit(self,
            input,
            **kwargs):
        return []


class DocsContextBuilder(ContextBuilder):

    def fit(self,
            docs: List[Dict],
            **kwargs):
        fitted_list = []
        for doc in docs:
            if count_tokens(fitted_list, **kwargs) + count_tokens(doc, **kwargs) < self.available_token_size:
                fitted_list.append(doc)
            else:
                break
        return fitted_list

