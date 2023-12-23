from tinyllm.eval.evaluator import Evaluator
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.llms.lite_llm_stream import LiteLLMStream
from tinyllm.util.helpers import get_openai_message

import asyncio

loop = asyncio.get_event_loop()

#### Basic chat
message = get_openai_message(role='user',
                             content="Hi")
litellm_chat = LiteLLM()
response = loop.run_until_complete(litellm_chat(messages=[message]))


#### Chat with evaluator

class SuccessFullRunEvaluator(Evaluator):
    async def run(self, **kwargs):
        print('Evaluating...')
        return {
            "evals": {
                "successful_score": 1,
            },
            "metadata": {}
        }


litellm_chat = LiteLLM(run_evaluators=[SuccessFullRunEvaluator()],
                       processed_output_evaluators=[SuccessFullRunEvaluator()])
message = get_openai_message(role='user',
                             content="Hi")
result = loop.run_until_complete(litellm_chat(messages=[message]))

#### Chat stream

litellmstream_chat = LiteLLMStream(name='Test: LiteLLM Stream')

async def get_stream():
    message = get_openai_message(role='user',
                                 content="Hi")
    async for msg in litellmstream_chat(messages=[message]):
        i = 0
    return msg


result = loop.run_until_complete(get_stream())
print(result['output']['streaming_status'])
