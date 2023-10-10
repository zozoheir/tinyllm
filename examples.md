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


#### QA Set Generator

```python
import asyncio

loop = asyncio.get_event_loop()

docs = [
    {"text":"Fake content 1"},
    {"text":"Fake context 2"}
]

qa_set_generator = QASetGenerator(
    name="QA Data Point Generator",
)
test_data_points = loop.run_until_complete(qa_set_generator(documents=docs,n=1))

```