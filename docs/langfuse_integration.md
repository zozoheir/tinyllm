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


Basic Example: Retrieval Chain

Below is an example of tracing a retrieval chain within a tinyllm application, showcasing the simplicity of integrating Langfuse tracing.
Typically, in tinyllm chains are created through the Function class, which is a class representation of a chain, allowing
for attributes, different methods etc. Functions are, also automatically traced.

```
TRACE: retrieval
|
|-- SPAN: vector_db_search
|-- GENERATION: user_output
|-- EVENT: db_insert
```
```python
import asyncio
from tinyllm.tracing.langfuse_context import observation

@observation('span')
async def retrieval(**kwargs):
    await vector_db_search(input=kwargs['input'])
    response = await user_output(messages=kwargs['messages'])
    await db_insert(input=kwargs['input'])
    return response

@observation('span')
async def vector_db_search(**kwargs):
    # Logic for vector database search
    return {'search_result': 'dummy_response'}

@observation('event')
async def db_insert(**kwargs):
    # Logic for database insert
    return {'db_status': 'success'}

@observation('generation')
async def user_output(**kwargs):
    # Logic for user output generation
    return {'message': 'User response'}

if __name__ == "__main__":
    asyncio.run(retrieval(input={'dummy_input'}, messages=[{'dummy_message'}]))

```



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
