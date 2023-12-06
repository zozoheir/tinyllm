from tinyllm.functions.lite_llm.lite_llm_chat import LiteLLM, LiteLLMStream


class LLMStore:
    LLMS = {
        "lite_llm": LiteLLM,
        "lite_llm_stream": LiteLLMStream,
    }

    def __init__(self,
                 tool_store):
        self.tool_store = tool_store

    def get_agent(self,
                  llm,
                  llm_args,
                  **kwargs):
        return LLMStore.LLMS[llm](**llm_args,
                                  **kwargs)
