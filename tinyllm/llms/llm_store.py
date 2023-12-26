from enum import Enum

from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.llms.lite_llm_stream import LiteLLMStream


class LLMs(Enum):
    LITE_LLM = LiteLLM
    LITE_LLM_STREAM = LiteLLMStream

class LLMStore:

    def get_llm(self, llm_library, **kwargs):
        llm_class = llm_library.value
        instance_of_class = llm_class(**kwargs)
        return instance_of_class
