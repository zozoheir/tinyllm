import asyncio

from tinyllm import default_embedding_model, langfuse_client
from tinyllm.functions.agent.agent_stream import AgentStream
from tinyllm.functions.agent.toolkit import Toolkit

from tinyllm.functions.agent.tool import Tool
from tinyllm.functions.eval.evaluator import Evaluator
from tinyllm.functions.examples.example_manager import ExampleManager
from tinyllm.functions.examples.example_selector import ExampleSelector
from tinyllm.functions.llms.llm_store import LLMStore, LLMs

examples = [[{
    "user": "Example question",
    "assistant": "Example answer",
},
    {
        "user": "Another example question",
        "assistant": "Another example answer"
    }
]]
example_selector = ExampleSelector(
    name="Test local example selector",
    examples=[],
    embedding_function=default_embedding_model,
    is_traced=False
)
example_manager = ExampleManager(
    constant_examples=[],
    example_selector=example_selector,
)


class AnswerCorrectnessEvaluator(Evaluator):

    async def run(self, **kwargs):
        response = kwargs['output']['response']
        evals = {
            "evals": {
                "correct_answer": 1 if 'january' in response['choices'][0]['message']['content'].lower() else 0
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
        is_traced=False,
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
    tools=tools,
)

llm_store = LLMStore()
manager_llm = llm_store.get_llm_function(
    llm_library=LLMs.LITE_LLM_STREAM,
    system_role="You are a helpful agent that can answer questions about the user's profile using available tools.",
    name='Tinyllm manager',
    example_manager=example_manager,
    is_traced=False,
    evaluators=[
        AnswerCorrectnessEvaluator(
            name="Answer Correctness Evaluator",
            is_traced=False,
        ),
    ]

)

tiny_agent = AgentStream(name='Example stream agent',
                         manager_llm=manager_llm,
                         toolkit=toolkit,
                         debug=True)


async def run_agent():
    msgs = []
    async for msg in tiny_agent(user_input="What is the user's birthday?"):
        msgs.append(msg)
        print(msg)
    return msgs


# Run
result = asyncio.run(run_agent())
print(result)
