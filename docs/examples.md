#### OpenAI chat

```python

import asyncio

from tinyllm.llms import OpenAIChat
from tinyllm.llms import OpenAIPromptTemplate

prompt_template = OpenAIPromptTemplate(
    name="TinyLLM Agent Prompt Template",
)
openai_chat = OpenAIChat(name='OpenAI Chat model',
                         model='gpt-3.5-turbo',
                         temperature=0,
                         rag_example
                         max_tokens=100,
                         prompt_template=prompt_template)

loop = asyncio.get_event_loop()
response = loop.run_until_complete(openai_chat(message='Hi how are you?'))
print(response)
```

#### OpenAI agent

```python
import asyncio

from tinyllm.llms import OpenAIChatAgent

test_openai_functions = [
    {
        "name": "test_function",
        "description": "Your default tool. This is the tool you use retrieve ANY information about the user. Use it to answer questions about his birthday and any personal info",
        "parameters": {
            "type": "object",
            "properties": {
                "asked_property": {
                    "type": "string",
                    "description": "The property the user asked about",
                },
            },
            "unit": {"type": "string", "enum": ["birthday", "name"]},
            "required": ["asked_property"],
        },
    }
]


def test_function(asked_property):
    if asked_property == "name":
        return "Elias"
    elif asked_property == "birthday":
        return "January 1st"


function_callables = {'test_function': test_function}

openai_agent = OpenAIChatAgent(
    name="Test TinyLLM Agent",
    model="gpt-3.5-turbo",
    openai_functions=test_openai_functions,
    function_callables=function_callables,
    temperature=0,
    max_tokens=1000,
    with_memory=True,
)

loop = asyncio.get_event_loop()
result = loop.run_until_complete(openai_agent(message="Oh nana...what's my name?"))
```

#### Context builder

```python
string_list = [
    {"content": "Fake content 1", "summary": "Fake summary 1", "title": "Fake title 1"},
    {"content": "Fake content 2", "summary": "Fake summary 2", "title": "Fake title 2"},
    {"content": "Fake content 3", "summary": "Fake summary 3", "title": "Fake title 3"}
]

context_builder = SingleSourceDocumentsFolderFitter(start_string="KNOWLEDGE GRAPH",
                                                    end_string="KNOWLEDGE GRAPH",
                                                    available_token_size=1000)
final_context = context_builder.to_string(
    search_results=string_list,
    header="[post]",
    ignore_keys=["summary", "title"]
)
print(final_context)
```

#### Evaluation pipeline

```python

import asyncio

loop = asyncio.get_event_loop()

kg_qa_chain = Retriever(
    name="KG QA Chain",
    rag_example
)


async def rag_lambda(input):
    chain_response = await kg_qa_chain(input=input)
    return chain_response['response']


# Generate test set
docs = [
    {"text": "Fake content 1"},
    {"text": "Fake context 2"}
]
qa_set_generator = QASetGenerator(
    name="QA Data Point Generator",
)
qa_test_set = loop.run_until_complete(qa_set_generator(documents=docs,
                                                       n=2))

# Initialize evaluators
answer_truth_evaluator = AnswerCorrectnessEvaluator(
    name="Answer Accuracy Evaluator",
)

# Initialize pipeline
eval_pipeline = RagEvaluationPipeline(
    rag_lambda=rag_lambda,
    qa_test_set=qa_test_set['qa_test_set'],
    evaluators=[answer_truth_evaluator]
)

evals = loop.run_until_complete(eval_pipeline.run_evals())
evals_df = pd.DataFrame(evals)


```
