"""
TRACE: rag_example
|   |   |
|   |-- SPAN: retrieval
|   |   |-- GENERATION: query_generation
|   |   |-- SPAN: vector_db_search
|   |   |
|   |-- GENERATION: final_answer
|   |   |
|   |-- SPAN: db_insert
|   |   |
|   |   |
"""


import asyncio

from tinyllm.tracing.langfuse_context import observation
from tinyllm.util.helpers import get_openai_message

# Dummy data for demonstration purposes
dummy_messages = [get_openai_message(role='user', content='dummy'),
                  get_openai_message(role='assistant', content='dummy')]
dummy_input = {}
dummy_response = {'dummy': 'dummy'}

@observation('span')
async def rag_example(**kwargs):
    # This is the top-level trace
    await retrieval(input=dummy_input)  # Retrieval span
    response = await final_answer(messages=dummy_messages)  # Final answer generation
    await db_insert(input=dummy_input)  # Database insert span
    return response

@observation('span')
async def retrieval(**kwargs):
    await vector_db_search(input=dummy_input)  # Vector database search event
    response = await user_output(messages=dummy_messages)  # User output generation
    return response

@observation('event')
async def vector_db_search(**kwargs):
    return dummy_response

@observation('event')
async def db_insert(**kwargs):
    return dummy_response

@observation('generation')
async def user_output(**kwargs):
    return {'message': get_openai_message(role='assistant', content='dummy')}

@observation('generation')
async def final_answer(**kwargs):
    return {'message': get_openai_message(role='assistant', content='dummy')}

if __name__ == "__main__":
    asyncio.run(rag_example(input=dummy_input))
