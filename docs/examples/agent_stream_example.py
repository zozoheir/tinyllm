import asyncio

from tinyllm.agent.agent_stream import AgentStream
from tinyllm.agent.tool import Tool
from tinyllm.agent.toolkit import Toolkit
from tinyllm.eval.evaluator import Evaluator
from tinyllm.llms.llm_store import LLMStore, LLMs

loop = asyncio.get_event_loop()

class AnswerCorrectnessEvaluator(Evaluator):
    async def run(self, **kwargs):
        completion = kwargs['output']['completion']
        evals = {
            "evals": {
                "correct_answer": 1 if 'january 1st' in completion.lower() else 0
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

async def run_agent_stream():

    llm_stream = llm_store.get_llm(
        llm_library=LLMs.LITE_LLM_STREAM,
        name='Tinyllm manager',
    )

    tiny_agent = AgentStream(name='Test: agent stream',
                             system_role="You are a helpful agent that can answer questions about the user's profile using available tools.",
                             llm=llm_stream,
                             toolkit=toolkit,
                             run_evaluators=[
                                 AnswerCorrectnessEvaluator(
                                     name="Functional call corrector",
                                 ),
                             ])

    msgs = []
    async for message in tiny_agent(user_input="What is the user's birthday?"):
        msgs.append(message)
    return msgs

result = loop.run_until_complete(run_agent_stream())
