from enum import Enum

from tinyllm.functions.llms.lite_llm import LiteLLM
from tinyllm.functions.llms.lite_llm_stream import LiteLLMStream


class LLMs(Enum):
    LITE_LLM = LiteLLM
    LITE_LLM_STREAM = LiteLLMStream

class LLMStore:

    def get_llm_function(self, llm_library, **kwargs):
        llm_class = llm_library.value
        return llm_class(**kwargs)
