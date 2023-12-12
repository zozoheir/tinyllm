import asyncio

from tinyllm.functions.agent.agent import Agent
from tinyllm.functions.agent.toolkit import Toolkit
from tinyllm.functions.llm.llm_store import LLMStore, LLMs

from tinyllm.functions.agent.tool import Tool
from tinyllm.functions.eval.evaluator import Evaluator


class AnswerCorrectnessEvaluator(Evaluator):

    async def run(self, **kwargs):
        evals = {
            "evals": {
            },
            "metadata": {}
        }
        completion = kwargs['output']['completion']
        if kwargs['processed_output']['type'] == 'tool':
            evals = {
                "evals": {
                    "functional_call": 1 if completion['name'] == 'get_user_property' else 0,
                },
                "metadata": {}
            }

        return evals


def get_user_property(asked_property):
    if asked_property == "name":
        return "Elias"
    elif asked_property == "birthday":
        return "January 1st"


tools = [
    Tool(
        name="get_user_property",
        description="This is the tool you use retrieve ANY information about the user. Use it to answer questions about his birthday and any personal info",
        python_lambda=get_user_property,
        parameters={
            "type": "object",
            "properties": {
                "asked_property": {
                    "type": "string",
                    "enum": ["birthday", "name"],
                    "description": "The specific property the user asked about",
                },
            },
            "required": ["asked_property"],
        },
    )
]
toolkit = Toolkit(
    name='Toolkit',
    tools=tools
)

llm_store = LLMStore()
llm_function = llm_store.get_llm_function(
    llm_library=LLMs.LITE_LLM_STREAM,
    system_role="You are a helpful agent that can answer questions about the user's profile using available tools.",
    name='Tinyllm manager',
    debug=False,
    evaluators=[
        AnswerCorrectnessEvaluator(
            name="Answer Correctness Evaluator",
            is_traced=False,
        ),
    ]
)

tiny_agent = Agent(name='TinyLLM Agent',
                   manager_function=llm_function,
                   toolkit=toolkit,
                   debug=True)


async def run_agent():
    msgs = []
    async for message in tiny_agent(user_input="What is the user's birthday?"):
        msgs.append(message)
    return msgs

# Run the asynchronous test
result = asyncio.run(run_agent())
print(result[-1])