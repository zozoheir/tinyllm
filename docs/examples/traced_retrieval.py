


########### EXAMPLE USAGE ##############
import asyncio

from tinyllm.tracing.langfuse_context import observation
from tinyllm.util.helpers import get_openai_message


@observation(type='span', name="search", input_mapping={'input': 'search_query'}, output_mapping={'output': 'docs', 'metadata': 'docs_metadata'})
async def search(**kwargs):
    return {
        'docs': [],
        'docs_metadata': {}
    }

@observation(type='generation', name="Food generate")
async def generate_food_msg(**kwargs):
    return {
        'message':get_openai_message(role='assistant', content='Moroccan food is delicious! I love it! Some of my favorite dishes are couscous, tagine, and harira.'),
    }

@observation(type='generation', name="Sports generate")
async def generate_sports_msg(**kwargs):
    return {
        'message':get_openai_message(role='assistant', content='Morocco is the first African team to ever qualify for World Cup semi finals'),
    }


@observation(type='span', name="Food retriever")
async def run_food_rag(**kwargs):
    result = await search(search_query=kwargs['search_query'])
    messages = [
        get_openai_message(
            role='assistant',
            content=str(result['docs']),
        )
    ]
    result = await generate_food_msg(messages=messages)
    return result

@observation(type='span', name="Sports retriever")
async def run_sports_rag(**kwargs):
    result = await search(search_query=kwargs['search_query'])
    messages = [
        get_openai_message(
            role='assistant',
            content=str(result['docs']),
        )
    ]
    result = await generate_sports_msg(messages=messages)
    return result


async def main():

    tasks = [
        asyncio.create_task(run_food_rag(search_query='Moroccan food')),
        asyncio.create_task(run_sports_rag(search_query='Moroccan food')),
    ]

    await asyncio.gather(*tasks)

# Run the main function in asyncio event loop
asyncio.run(main())
