"""
QuestionAnswerGenerator:
- input: documents
- output: list of (context, question, answer, context_answer_similarity) dicts

"""

import random
import re
from textwrap import dedent
from typing import List

from env_util.supabase import supabase_client
from tinyllm.functions.function import Function
from tinyllm.functions.llms.open_ai.util.helpers import get_user_message, get_assistant_message
from tinyllm.functions.validator import Validator
from tinyllm.functions.llms.open_ai.openai_chat import OpenAIChat
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate


class InputQASetGenerator(Validator):
    documents: list
    n: int


class OutputQASetGenerator(Validator):
    chat_responses: List[str]


INPUT_EXAMPLE = """
The spate of recommendations ended the silent period for the nearly 30 banks that underwrote Arm’s IPO in September.
The chip manufacturer raised $4.87 billion for its owner, SoftBank Group, marking the biggest public listing of 2023. From a broader perspective, the IPO’s success provided much-needed confidence to investors and companies considering going public following a nearly two-year market drought. Arm’s IPO was one of the three big September listings, with delivery company Instacart and marketing automation firm Klaviyo debuting on the US stock exchanges.
With Arm’s shares currently trading at $55.5 a piece, the aforementioned price targets by Wall Street giants imply the stock has an upside potential of between 10% and 27%. Meanwhile, some brokerages, like HSBC, offered a more cautious coverage for Arm’s stock, saying the company’s shares may remain range-bound due to smartphone market uncertainty.
Where do you think Arm’s share price will stand by the end of 2023? Let us know in the comments below.
"""
OUTPUT_EXAMPLE = """
Question: How much did Arm raise for its owner, SoftBank Group, during its IPO?
Answer: Arm Holdings raised $4.87 billion for its owner, SoftBank Group, during its IPO.
"""

example_msgs = [
    get_user_message(INPUT_EXAMPLE),
    get_assistant_message(OUTPUT_EXAMPLE)
]
qa_prompt_template = OpenAIPromptTemplate(
    name="QA Data Point Generator Template",
    system_role=dedent(f"""
ROLE:
You are a knowledgeable expert. Given a context, your role is to generate a relevant question about the context and 
provide a truthful answer based on the information in the context.
"""),
    messages=example_msgs,
    is_traced=False
)


class QASetGenerator(Function):

    def __init__(self, **kwargs):
        super().__init__(input_validator=InputQASetGenerator,
                         output_validator=OutputQASetGenerator,
                         **kwargs)

        self.openai_chat = OpenAIChat(
            name="QA Data Point Generator",
            llm_name='gpt-3.5-turbo',
            max_tokens=600,
            prompt_template=qa_prompt_template,
            is_traced=True,
            llm_trace=self.llm_trace,
            with_memory=False,
        )

    async def run(self, **kwargs):
        documents = kwargs["documents"]
        n = kwargs["n"]

        chat_responses = []
        for _ in range(n):
            context = random.choice(documents)["text"]
            openai_response = await self.openai_chat(
                message=context,
                generation_name="QA Data Point Generator")
            chat_responses.append(openai_response['response'])

        return {"chat_responses": chat_responses}

    async def process_output(self, **kwargs) -> dict:
        chat_responses = kwargs["chat_responses"]
        parsed_test_data_points = []
        for test_data_point in chat_responses:
            question_match = re.search(r"Question: (.+?)\n", test_data_point)
            answer_match = re.search(r"Answer: (.+)", test_data_point)

            if not (question_match and answer_match):
                raise ValueError("The provided string doesn't match the expected format.")
            else:
                parsed_test_data_points.append({
                    "question": question_match.group(1).strip(),
                    "answer": answer_match.group(1).strip()
                })
        return {"test_data_points": parsed_test_data_points}

