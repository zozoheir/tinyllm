#### Context builder

```python
string_list = [
    {"content": "Fake content 1", "summary": "Fake summary 1", "title": "Fake title 1"},
    {"content": "Fake content 2", "summary": "Fake summary 2", "title": "Fake title 2"},
    {"content": "Fake content 3", "summary": "Fake summary 3", "title": "Fake title 3"}
]

context_builder = SummaryContextBuilder(start_string="SUPPORTING DOCS",
                                        end_string="SUPPORTING DOCS",
                                        available_token_size=1000)
final_context = context_builder.get_context(
    docs=string_list,
    header="[post]",
    ignore_keys=["summary", "title"]
)
print(final_context)
```