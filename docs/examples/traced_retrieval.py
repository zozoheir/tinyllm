import asyncio

from tinyllm.tracing.langfuse_context import observation
from tinyllm.util.helpers import get_openai_message


@observation(type='span', name="search", input_mapping={'input': 'search_query'}, output_mapping={'output': 'docs', 'metadata': 'docs_metadata'})
async def search(**kwargs):
    return {
        'docs': [],
        'docs_metadata': {}
    }

@observation(type='generation', name="generate")
async def generate(**kwargs):
    return {
        'message':get_openai_message(role='assistant', content='Moroccan food is delicious! I love it! Some of my favorite dishes are couscous, tagine, and harira.'),
    }

@observation(type='span', name="Retriever")
async def run_retriever(**kwargs):
    result = await search(search_query=kwargs['search_query'])
    messages = [
        get_openai_message(
            role='assistant',
            content=str(result['docs']),
        )
    ]
    result = await generate(messages=messages)
    return result


async def main():
    await run_retriever(search_query='Moroccan food')

# Run the main function in asyncio event loop
asyncio.run(main())
