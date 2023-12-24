# tinyllm Langfuse Integration Documentation
## Overview

Integrating Langfuse with tinyllm provides a streamlined and efficient
way to trace and monitor Large Language Model (LLM) applications. 
tinyllm, known for its simplicity and readability, now extends its 
capabilities with Langfuse tracing, allowing for detailed and 
insightful monitoring of LLM functions, agents, and models.

The integration primarily revolves around the @observation 
decorator. This decorator simplifies the process 
of tracing various components of LLM applications, ensuring 
that every action and outcome within these applications is 
automatically logged and monitored via Langfuse.

## Async and Streaming support
The tinyllm integration supports Async functions and generators for streaming mode.
In streaming mode, the observation decorator will use the last message yielded by 
the generator for closing the observation. 

## Observations nesting

All nested decorated calls are automatically logged as a tree in 
Langfuse. You can easily create and trace various and complex
compute graphs.

## Usage


[Basic Example: Retrieval Chain
](https://github.com/zozoheir/tinyllm/blob/main/docs/examples/tracing_nested_functions.py): check out this example of tracing a retrieval chain of functions within a tinyllm application, showcasing the simplicity of integrating Langfuse tracing.

While this example traces python function calls, typically in tinyllm chains are using through the Function class, allowing
for class attributes/extensions/customizations, different methods etc. 


## Mapping function inputs and outputs to Langfuse fields

Use the optional input_mapping and output_mapping observation kwargs 
to map, using a dictionary, which of your functions keyword arguments 
should be mapped to which Langfuse fields. 
Example here to map the input argument in Langfuse to the function's 
"messages" keyword argument.
```python
@observation('generation', 
             input_mapping={'input': 'messages'},
             outcome_mapping={'output': 'answer'})
async def generate_answer(**kwargs):
    messages = kwargs['messages']
    return {'answer': 'dummy_answer'}
```
