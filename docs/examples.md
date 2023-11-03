#### OpenAI chat

```python

import asyncio

from tinyllm.functions.llms.open_ai.openai_chat import OpenAIChat
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate

prompt_template = OpenAIPromptTemplate(
    name="TinyLLM Agent Prompt Template",
)
openai_chat = OpenAIChat(name='OpenAI Chat model',
                         model='gpt-3.5-turbo',
                         temperature=0,
                         is_traced=True,
                         max_tokens=100,
                         prompt_template=prompt_template)

loop = asyncio.get_event_loop()
response = loop.run_until_complete(openai_chat(message='Hi how are you?'))
print(response)
```

#### OpenAI agent
```python
import asyncio

from tinyllm.functions.llms.open_ai.openai_chat_agent import OpenAIChatAgent

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

context_builder = SingleSourceDocsContextBuilder(start_string="SUPPORTING DOCS",
                                        end_string="SUPPORTING DOCS",
                                        available_token_size=1000)
final_context = context_builder.get_context(
    docs=string_list,
    header="[post]",
    ignore_keys=["summary", "title"]
)
print(final_context)
```


#### Evaluation pipeline


```python

import asyncio
loop = asyncio.get_event_loop()

kg_qa_chain = KGQAChain(
    name="KG QA Chain",
    is_traced=True,
)

async def rag_lambda(user_question):
    chain_response = await kg_qa_chain(user_question=user_question)
    return chain_response['response']

# Generate test set
docs = [
    {"text":"Fake content 1"},
    {"text":"Fake context 2"}
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


#### Credit analysis chain

```python
import asyncio

from tinyllm.functions.chain import Chain
from tinyllm.functions.decision import Decision
from tinyllm.functions.llms.open_ai.openai_chat import OpenAIChat
from tinyllm.functions.concurrent import Concurrent
from tinyllm.functions.function import Function
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate

good_loan_application_example = """
The loan application showcases a commendable financial profile with an excellent credit history. The applicant's credit score
demonstrates a history of timely payments, responsible credit usage, and a low utilization rate, reflecting a consistent
track record of financial prudence. Moreover, the applicant's income documentation reveals a stable employment history
with a steady and substantial income stream. The debt-to-income ratio is well within the acceptable range, indicating a
manageable level of existing debt. Overall, the applicant's solid credit standing and stable financial situation make
this loan application an appealing opportunity for potential lenders.
"""

loan_classifier_prompt_template = OpenAIPromptTemplate(
    name="KG Extractor prompt template",
    system_role="Your role is to classify if as as good or bad. Your output should be one one of these 2 words:[good, bad]",
)

openai_chat = OpenAIChat(name='OpenAI-GPT model',
                         model='gpt-3.5-turbo',
                         temperature=0,
                         n=1,
                         prompt_template=loan_classifier_prompt_template)


# Loan classifier LLM
async def classify_loan_application(**kwargs):
    loan_application = kwargs.get('loan_application')
    chat_response = await openai_chat(message=loan_application)
    print(f"Credit classification: {chat_response}")
    return {'decision': chat_response['response']}


loan_classifier = Decision(
    name="Decision: Loan classifier",
    choices=['good', 'bad'],
    run_function=classify_loan_application,
    is_traced=True
)


# Email notification
async def send_email(**kwargs):
    print("Sending email notification...")
    return {'success': True}


email_notification = Function(
    name="Email notification",
    run_function=send_email
)


# Background get_check_results
async def background_check(**kwargs):
    print("Performing background get_check_results")
    return {'background_check': 'Completed'}


bg_check = Function(
    name="Background get_check_results",
    run_function=background_check,
)


# Chain
async def main():
    credit_analysis_chain = Chain(
        name="Chain: Loan application",
        children=[
            loan_classifier,
            Concurrent(name="Concurrent: On good credit",
                       children=[
                           email_notification,
                           bg_check])
        ],
        is_traced=True)

    result = await credit_analysis_chain(loan_application=good_loan_application_example)


asyncio.run(main())

```