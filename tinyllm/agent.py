import asyncio
from typing import List

import gradio as gr
import os
import openai
from langchain.vectorstores import PGVector

from tinyllm.cache.cache import LocalDirCache
from tinyllm.functions.function import Function
from tinyllm.functions.llms.openai.openai_chat import OpenAIChat
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.vector_store import get_vector_store

openai.api_key = os.getenv("OPENAI_API_KEY")

loop = asyncio.get_event_loop()

prompt_template = OpenAIPromptTemplate(
    name="TinyLLM Agent Prompt Template",
    system_role="You help user with their question.",
    messages=[
        "You are a world-class Python developer and expert of the tinyllm library code, documentation and logic.",
        "You help users understand tinyllm and build chains, llm workflows and all types of tinyllm functions.",
    ],
)

tinyllm_vector_collection = get_vector_store(collection_name="tinyllm")

functions = [
    {
        "name": "get_tinyllm_supporting_docs",
        "description": "Get tinyllm Python code and documentation",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Rephrased user question to find relevant tinyllm Python code and documentation",
                },
            },
            "required": ["input"],
        },
    }
]

openai_chat = OpenAIChat(
    name="TinyLLM Agent",
    llm_name="gpt-3.5-turbo-16k",
    temperature=0,
    prompt_template=prompt_template,
    n=1,
    functions=functions,
    verbose=True,
)

local_file_cache = LocalDirCache(
    collection_name="tinyllm",
    directory_name="/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm",
    vector_collection=tinyllm_vector_collection,
)
local_file_cache.refresh_cache()


def get_tinyllm_supporting_docs(message):
    search_results = tinyllm_vector_collection.similarity_search(message, n=3)
    files = []
    function_class = local_file_cache.get_file_content(
        "/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/functions/function.py"
    )
    files.append(function_class)
    files += [i.page_content for i in search_results]
    return "\n".join(files)

tinyllm_agent_tools = {
    'get_tinyllm_supporting_docs': get_tinyllm_supporting_docs
}


def tinyllm_chat(message, history):

    # Get the response from the AI agent
    chat_response = loop.run_until_complete(openai_chat(message=message))
    return chat_response["response"]


CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 80vh !important; }
#component-0 { height: 80%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""
demo = gr.ChatInterface(
    tinyllm_chat,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(
        placeholder="Ask any question", container=True, scale=15, lines=6
    ),
    title="Tinyllm AI Agent",
    examples=["How do I build a chain with tinyllm?"],
    cache_examples=False,
    retry_btn="Try Again",
    undo_btn=None,
    clear_btn="Clear",
)
demo.launch(debug=True, inbrowser=False)
