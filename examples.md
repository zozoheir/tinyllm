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
answer_truth_evaluator = AnswerTruthfulnessEvaluator(
    name="Answer Accuracy Evaluator",
)

# Initialize pipeline
eval_pipeline = EvaluationPipeline(
    rag_lambda=rag_lambda,
    qa_test_set=qa_test_set['qa_test_set'],
    evaluators=[answer_truth_evaluator]
)

evals = loop.run_until_complete(eval_pipeline.run_evals())
evals_df = pd.DataFrame(evals)


```