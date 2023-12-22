import random
import re
from textwrap import dedent
from typing import List

from tinyllm.function import Function
from tinyllm.llms.lite_llm import LiteLLM
from tinyllm.util.helpers import get_openai_message
from tinyllm.validator import Validator

INPUT_EXAMPLE = """
Relevant context for question/answer generation:
The spate of recommendations ended the silent period for the nearly 30 banks that underwrote Arm’s IPO in September.
The chip manufacturer raised $4.87 billion for its owner, SoftBank Group, marking the biggest public listing of 2023. 
From a broader perspective, the IPO’s success provided much-needed confidence to investors and companies considering 
going public following a nearly two-year market drought. Arm’s IPO was one of the three big September listings, with 
delivery company Instacart and marketing automation firm Klaviyo debuting on the US stock exchanges.
With Arm’s shares currently trading at $55.5 a piece, the aforementioned price targets by Wall Street giants imply the stock has an upside potential of between 10% and 27%. Meanwhile, some brokerages, like HSBC, offered a more cautious coverage for Arm’s stock, saying the company’s shares may remain range-bound due to smartphone market uncertainty.
Where do you think Arm’s share price will stand by the end of 2023? Let us know in the comments below.
"""
OUTPUT_EXAMPLE = """
Question: How much did Arm raise for its owner, SoftBank Group, during its IPO?
Truthful answer: Arm raised $4.87 billion for its owner, SoftBank Group, during its IPO.
"""

example_msgs = [
    get_openai_message(role='user', content=INPUT_EXAMPLE),
    get_openai_message(role='assistant', content=OUTPUT_EXAMPLE)
]
system_role=dedent(f"""
ROLE:
You are a knowledgeable expert. Given a context, your role is to generate a relevant question about the context and 
provide a truthful answer based on the information in the context.
""")

class InputQASetGenerator(Validator):
    documents: list
    n: int

class OutputQASetGenerator(Validator):
    qa_test_set: List[dict] # dict with keys: context, chat_response, question, correct_answer


class QASetGenerator(Function):

    def __init__(self,
                 **kwargs):
        super().__init__(input_validator=InputQASetGenerator,
                         output_validator=OutputQASetGenerator,
                         **kwargs)

        self.openai_chat = LiteLLM(
            name="QA Data Point Generator",
            system_role=system_role,
        )

    async def run(self, **kwargs):
        documents = kwargs["documents"]
        n = kwargs["n"]

        qa_test_set = []
        for _ in range(n):
            doc = random.choice(documents)
            context = doc["text"]
            formatted_context = dedent(f"""
            Relevant context for question/answer generation:
            {context}
            """)
            openai_response = await self.openai_chat(
                message=formatted_context,
                generation_name="QA Data Point Generator"
            )

            qa_test_set.append({
                "metadata": doc["emetadata"],
                "truth_context": context,
                "chat_response": openai_response['output']["response"]
            })

        return {"qa_test_set": qa_test_set}

    async def process_output(self, **kwargs) -> dict:
        qa_test_set = kwargs["qa_test_set"]
        for test_data_point in qa_test_set:
            chat_response = test_data_point['chat_response']
            question_match = re.search(r"Question: (.+?)\n", chat_response)
            answer_match = re.search(r"Truthful answer: (.+)", chat_response)

            if not (question_match and answer_match):
                raise ValueError("The provided string doesn't match the expected format.")
            else:
                question_match = question_match.group(1).strip()
                answer_match = answer_match.group(1).strip()
                if question_match and answer_match:
                    test_data_point.update({
                        "question": question_match,
                        "correct_answer": answer_match
                    })
                else:
                    raise ValueError("The provided string doesn't match the expected format.")
        return {"qa_test_set": qa_test_set}