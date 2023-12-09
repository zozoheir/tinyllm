from typing import List, Dict

from tinyllm.functions.rag.context_builder import ContextBuilder
from tinyllm.functions.helpers import get_user_message, count_tokens
from tinyllm.functions.rag.document import Document
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
            docs: List[Document]):
        fitted_doc_list = []
        for doc in docs:
            doc_list_size = sum([doc.size for doc in fitted_doc_list])+len('\n')*len(fitted_doc_list)
            if doc_list_size + doc.size < self.available_token_size:
                fitted_doc_list.append(doc)
            else:
                break
        return fitted_doc_list

    def get_context(self,
                    docs: List[Document]):
        docs = self.fit(docs=docs)
        fitted_context_string = '\n'.join([doc.format() for doc in docs])
        formatted_context = self.format_context(fitted_context_string)
        return formatted_context
