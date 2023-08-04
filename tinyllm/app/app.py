import asyncio

import gradio as gr
import os
import openai

from tinyllm.app.helpers import tinyllm_agent_functions, function_callables, tinyllm_agent_prompt_template, \
    local_file_cache
from tinyllm.functions.llms.open_ai.openai_chat_agent import OpenAIChatAgent

openai.api_key = os.getenv("OPENAI_API_KEY")
loop = asyncio.get_event_loop()
openai_agent = OpenAIChatAgent(
    name="TinyLLM Agent",
    llm_name="gpt-3.5-turbo",
    openai_functions=tinyllm_agent_functions,
    function_callables=function_callables,
    temperature=0,
    prompt_template=tinyllm_agent_prompt_template,
    max_tokens=1000,
    tracing=True,
)

chat_response = loop.run_until_complete(openai_agent(message="Help me write a unit test for the tinyllm chat agent class"))


def tinyllm_chat(message, history):
    # Get the response from the AI agent
    chat_response = loop.run_until_complete(openai_agent(message=message))
    print(chat_response)
    return chat_response["response"]


local_file_cache.refresh_cache()


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
