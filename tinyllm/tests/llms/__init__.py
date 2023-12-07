import asyncio

from tinyllm.functions.lite_llm.lite_llm import LiteLLM

loop = asyncio.get_event_loop()
litellm_chat = LiteLLM(name='Test: LiteLLMChat',
                       with_memory=True)
result = loop.run_until_complete(litellm_chat(content="Hi"))
