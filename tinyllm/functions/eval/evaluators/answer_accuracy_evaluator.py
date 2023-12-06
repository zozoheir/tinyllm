import re
from textwrap import dedent

from tinyllm.functions.eval.evaluator import Evaluator
from tinyllm.functions.llms.openai.openai_chat import OpenAIChat
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.lite_llm.util import get_user_message, get_assistant_message

EXAMPLE_INPUT = """
Context:
The spate of recommendations ended the silent period for the nearly 30 banks that underwrote Arm’s IPO in September.
The chip manufacturer raised $4.87 billion for its owner, SoftBank Group, marking the biggest public listing of 2023. From a broader perspective, the IPO’s success provided much-needed confidence to investors and companies considering going public following a nearly two-year market drought. Arm’s IPO was one of the three big September listings, with delivery company Instacart and marketing automation firm Klaviyo debuting on the US stock exchanges.
With Arm’s shares currently trading at $55.5 a piece, the aforementioned price targets by Wall Street giants imply the stock has an upside potential of between 10% and 27%. Meanwhile, some brokerages, like HSBC, offered a more cautious coverage for Arm’s stock, saying the company’s shares may remain range-bound due to smartphone market uncertainty.
Where do you think Arm’s share price will stand by the end of 2023? Let us know in the comments below.

Question: 
How much did Arm raise for its owner, SoftBank Group, during its IPO?

Correct answer:
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
- Correctness score: 5/10
"""

examples = [
    get_user_message(EXAMPLE_INPUT),
    get_assistant_message(EXAMPLE_OUTPUT)
]

accuracy_prompt_template = OpenAIPromptTemplate(
    name="Answer Accuracy Evaluation Template",
    system_role=dedent(f"""
ROLE:
You are an evaluator. Given a question, a correct answer, and a generated answer, you are to evaluate the correctness of the 
predicted answer on a scale of 0 to 10 with respect to the question asked and correct answer.
You will think and reason about the correctness of the generated answer then provide a Correctness score.
If the the generated answer is "Not enough information", the score should be 0.
"""),
    messages=examples,
    is_traced=False
)


class AnswerCorrectnessEvaluator(Evaluator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.openai_chat = OpenAIChat(
            name="Answer Accuracy Evaluator",
            model='gpt-3.5-turbo',
            max_tokens=400,
            prompt_template=accuracy_prompt_template,
            is_traced=True,
            llm_trace=self.trace,
            with_memory=False,
        )

    async def run(self, **kwargs):
        context = kwargs["retrieved_chunks"]
        question = kwargs["input"]
        correct_answer = kwargs["correct_answer"]
        generated_answer = kwargs["response"]
        formatted_message = f"""
Context:
{context}

Question:
{question}

Correct answer:
{correct_answer}

Generated answer:
{generated_answer}
        """
        openai_response = await self.openai_chat(
            message=formatted_message,
            generation_name="Answer Correctness Evaluator")
        chat_response = openai_response['output']["response"]
        # Regex patterns
        reasoning_pattern = r"- Reasoning: (.*?)- Correctness score:"
        correct_score_pattern = r"- Correctness score: (.*)"
        reasoning_match = re.search(reasoning_pattern, chat_response, re.DOTALL)
        truth_score_match = re.search(correct_score_pattern, chat_response)
        if reasoning_match:
            truth_score_reasoning = reasoning_match.group(1).strip()
        if truth_score_match:
            correctness_score = float(truth_score_match.group(1).strip().split('/')[0]) / 10
        return {
            "evals": {
                "correctness_score": correctness_score,
            },
            "metadata": {
                "truth_score_reasoning": truth_score_reasoning,
            }
        }
