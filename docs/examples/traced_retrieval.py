import asyncio

from tinyllm import langfuse_client
from tinyllm.tracing.langfuse_context import observation
from tinyllm.util.helpers import get_openai_message


@observation(observation_type='span', name="search", input_mapping={'input': 'search_query'}, output_mapping={'output': 'docs', 'metadata': 'docs_metadata'})
async def search(**kwargs):
    return {
        'docs': [],
        'docs_metadata': {}
    }

@observation(observation_type='generation', name="generate")
async def generate(**kwargs):
    return {
        'message':get_openai_message(role='assistant', content='Moroccan food is delicious! I love it! Some of my favorite dishes are couscous, tagine, and harira.'),
    }

@observation(observation_type='span', name="Sports retriever")
async def run_sports_retriever(**kwargs):
    result = await search(search_query=kwargs['search_query'])
    messages = [
        get_openai_message(
            role='assistant',
            content=str(result['docs']),
        )
    ]
    result = await generate(messages=messages)
    return {
        'message': 'Moroccan sports'
    }


async def main():
    tasks = [
        asyncio.create_task(run_sports_retriever(search_query='Moroccan sports')),
        asyncio.create_task(run_sports_retriever(search_query='Moroccan sports')),
    ]
    await asyncio.gather(*tasks)

# Run the main function in asyncio event loop
asyncio.run(main())
