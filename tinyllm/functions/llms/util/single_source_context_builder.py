from typing import List, Dict

from tinyllm.functions.llms.lite_llm.util.helpers import *
from tinyllm.functions.llms.util.context_builder import ContextBuilder
from tinyllm.util.prompt_util import stringify_dict_list

class SingleSourceDocsContextBuilder(ContextBuilder):
    def __init__(self,
                 start_string: str,
                 end_string: str,
                 available_token_size: int):
        self.start_string = start_string
        self.end_string = end_string
        self.available_token_size = available_token_size
        self.fitted_input = None
        self.context = None


    def fit(self,
            docs: List[Dict],
            **kwargs):
        fitted_list = []
        for doc in docs:
            if count_tokens(input=fitted_list, **kwargs) + count_tokens(input=doc, **kwargs) < self.available_token_size:
                fitted_list.append(doc)
            else:
                break
        return fitted_list


    def get_context(self,
                    docs,
                    output_format="str",
                    header="[doc]",
                    ignore_keys=[],
                    **kwargs):
        # If list of strings, convert to list of dicts
        if isinstance(docs, List) and isinstance(docs[0], str):
            docs = [{"content": string} for string in docs]

        # If list of dictionaries
        if isinstance(docs, List):
            fitted_list = self.fit(docs=docs,
                                   output_format="str",
                                   header="[doc]",
                                   ignore_keys=[],
                                   **kwargs)
            fitted_context_string = stringify_dict_list(header=header,
                                                        dicts=fitted_list,
                                                        ignore_keys=ignore_keys)

        formatted_context = self.format_context(fitted_context_string)
        if output_format == "openai":
            return get_user_message(formatted_context)
        elif output_format == "str":
            return formatted_context

