from typing import Dict

from smartpy.utility.log_util import getLogger
from tinyllm.functions.llms.open_ai.util.helpers import num_tokens_from_string
from tinyllm.util.prompt_util import stringify_dict_list, stringify_dict

logger = getLogger(__name__)


class LLMJobBatchSizeExceeded(Exception):
    pass


class OpenAIJobBatch:

    def __init__(self,
                 token_size_limit: int,
                 dict_header='[post]'):
        self.job_units_dicts = []
        self.token_size_limit = token_size_limit
        self.dict_header = dict_header

    def add(self,
            news_dict: Dict,
            ):
        dict_token_size = num_tokens_from_string(stringify_dict(header=self.dict_header, dict=news_dict))
        if dict_token_size + self.token_size < self.token_size_limit:
            self.job_units_dicts.append(news_dict)
        else:
            raise LLMJobBatchSizeExceeded(f"News dict size exceeds token size limit of {self.token_size_limit}.")

    @property
    def batch_prompt(self) -> str:
        return stringify_dict_list(dict_header=self.dict_header, dicts=self.job_units_dicts, ignore_keys=['source','author','timestamp'])

    @property
    def token_size(self) -> int:
        return num_tokens_from_string(self.batch_prompt)
