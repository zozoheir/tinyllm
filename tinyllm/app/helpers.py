import asyncio
import os
import openai

from tinyllm.cache.cache import LocalDirCache
from tinyllm.functions.llms.open_ai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.vector_store import get_vector_collection

openai.api_key = os.getenv("OPENAI_API_KEY")

loop = asyncio.get_event_loop()

tinyllm_vector_collection = get_vector_collection(collection_name="tinyllm")
local_file_cache = LocalDirCache(
    collection_name="tinyllm",
    directory_name="/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm",
    vector_collection=tinyllm_vector_collection,
)

tinyllm_agent_prompt_template = OpenAIPromptTemplate(
    name="TinyLLM Agent Prompt Template",
    system_role="""
    # ROLE
    You are the TinyLLM agent, a world-class Python developer and copilot for users of a new library, tinyllm. 

    # KNOWLEDGE
    You only have knowledge about the tinyLLM library through the search repository function and the results of function calls. You don't know ANYTHING else.
    
    # TASK:
    Answer any user question, and implement any code request using the tinyllm library. Use the function to find tinyllm code, examples, and documentation. 

    # FUNCTIONS
    You MUST only use one of the following functions:
    - search_repository: search tool for the tinyllm library code, documentation and logic through a function. 
    """,
    messages=[
    ],
)


tinyllm_agent_functions = [
    {
        "name": "search_repository",
        "description": "Your default tool. This is a search tool for the tinyllm code repository and documentation. Use this function to search "
                       "for any information about tinyllm, find examples on how to write chains, workflows, functions tests with tinyllm "
                       "and documentation about tinyllm.",
        "parameters": {
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "The user input to the tinyllm agent.",
                },
            },
            "required": ["search"],
        },
    }
]


def get_tinyllm_content(search):
    search_results = tinyllm_vector_collection.similarity_search(search, n=3)

    files = []
    function_class = local_file_cache.get_file_content(
        "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/functions/function.py"
    )
    files.append(function_class)
    files += [i.page_content for i in search_results]
    # Create the context
    supporting_docs = "\n".join(files)
    context = f"""
    Use the following documents to answer the question: 
    {supporting_docs}
    """
    final_message = context + "\n----------\n"
    return final_message


function_callables = {
    'search_repository': get_tinyllm_content
}
