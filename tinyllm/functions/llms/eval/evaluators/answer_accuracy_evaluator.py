import re
from textwrap import dedent

from tinyllm.functions.function import Function
from tinyllm.functions.llms.open_ai.openai_chat import OpenAIChat
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.llms.open_ai.util.helpers import get_user_message, get_assistant_message
from tinyllm.functions.validator import Validator



EXAMPLE_INPUT = """
Context:
The spate of recommendations ended the silent period for the nearly 30 banks that underwrote Arm’s IPO in September.
The chip manufacturer raised $4.87 billion for its owner, SoftBank Group, marking the biggest public listing of 2023. From a broader perspective, the IPO’s success provided much-needed confidence to investors and companies considering going public following a nearly two-year market drought. Arm’s IPO was one of the three big September listings, with delivery company Instacart and marketing automation firm Klaviyo debuting on the US stock exchanges.
With Arm’s shares currently trading at $55.5 a piece, the aforementioned price targets by Wall Street giants imply the stock has an upside potential of between 10% and 27%. Meanwhile, some brokerages, like HSBC, offered a more cautious coverage for Arm’s stock, saying the company’s shares may remain range-bound due to smartphone market uncertainty.
Where do you think Arm’s share price will stand by the end of 2023? Let us know in the comments below.

Question: 
How much did Arm raise for its owner, SoftBank Group, during its IPO?

Truthful answer:
Arm raised $4.87 billion for its owner, SoftBank Group, during its IPO.

Generated answer: 
Arm Holdings raised a couple of billion dollars for its owner, SoftBank Group, during its IPO.
"""

EXAMPLE_OUTPUT = """
- Reasoning: The generated answer states that Arm Holdings raised "a couple of billion dollars" for its owner, SoftBank Group, during 
its IPO. The context clearly states that the chip manufacturer (Arm) raised "$4.87 billion" for SoftBank Group. 
The phrase "a couple of billion dollars" is typically interpreted as meaning "two billion dollars", which is 
significantly less than $4.87 billion. While the generated answer is in the correct ballpark, it is imprecise and 
understates the actual amount by nearly $3 billion.
- Truth score: 5/10
"""

examples = [
    get_user_message(EXAMPLE_INPUT),
    get_assistant_message(EXAMPLE_OUTPUT)
]

accuracy_prompt_template = OpenAIPromptTemplate(
    name="Answer Accuracy Evaluation Template",
    system_role=dedent(f"""
ROLE:
You are an evaluator. Given a question, a truthful answer, and a generated answer, you are to evaluate the truthfulness of the 
predicted answer on a scale of 0 to 10. 
You will think and reason about the truthfullness of the generated answer then provide a truth score.
If the the generated answer is "Not enough information", the score should be 0.
"""),
    messages=examples,
    is_traced=False
)


class InputAnswerAccuracyEvaluator(Validator):
    context: str
    question: str
    truthful_answer: str
    generated_answer: str

class OutputAnswerAccuracyEvaluator(Validator):
    chat_response: str

class AnswerTruthfulnessEvaluator(Function):

    def __init__(self, **kwargs):
        super().__init__(input_validator=InputAnswerAccuracyEvaluator,
                         output_validator=OutputAnswerAccuracyEvaluator,
                         **kwargs)

        self.openai_chat = OpenAIChat(
            name="Answer Accuracy Evaluator",
            llm_name='gpt-3.5-turbo',
            max_tokens=400,
            prompt_template=accuracy_prompt_template,
            is_traced=True,
            llm_trace=self.llm_trace,
            with_memory=False,
        )

    async def run(self, **kwargs):
        context = kwargs["context"]
        question = kwargs["question"]
        truthful_answer = kwargs["truthful_answer"]
        generated_answer = kwargs["generated_answer"]
        formatted_message = f"""
Context:
{context}
        
Question:
{question}

Truthful answer:
{truthful_answer}

Generated answer:
{generated_answer}
        """
        openai_response = await self.openai_chat(
            message=formatted_message,
            generation_name="Answer Accuracy Evaluator")

        return {"chat_response": openai_response['response']}

    async def process_output(self, **kwargs) -> dict:
        chat_response = kwargs["chat_response"]
        # Regex patterns
        reasoning_pattern = r"- Reasoning: (.*?)- Truth score:"
        truth_score_pattern = r"- Truth score: (.*)"
        reasoning_match = re.search(reasoning_pattern, chat_response, re.DOTALL)
        truth_score_match = re.search(truth_score_pattern, chat_response)
        output_dict = {}
        if reasoning_match:
            output_dict["truth_score_reasoning"] = reasoning_match.group(1).strip()
        if truth_score_match:
            output_dict["truth_score"] = truth_score_match.group(1).strip()

        return output_dict

