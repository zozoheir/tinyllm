import asyncio

from tinyllm.tracing.langfuse_context import observation


# Example usage
@observation(type='span', name="search", input_mapping={'input': 'search_query'}, output_mapping={'output': 'docs', 'metadata': 'docs_metadata'})
async def search(**kwargs):
    # async search logic
    return {
        'docs': [],
        'docs_metadata': {}
    }
@observation(type='generation', name="generate")
async def generate(**kwargs):
    return {
        'agent_response':'Moroccan food is delicious! I love it! Some of my favorite dishes are couscous, tagine, and harira.'
    }

@observation(type='span', name="Retriever")
async def run_retriever(**kwargs):
    result = await search(**kwargs)
    await generate(**result)

# Running the async function
async def main():
    await run_retriever(search_query='Moroccan food')

# Run the main function in asyncio event loop
asyncio.run(main())
asyncio.run(main())
