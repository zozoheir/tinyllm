import asyncio
import gradio as gr
import os
import openai
from tinyllm.cache.cache import LocalFilesCache
from tinyllm.functions.llms.openai.openai_chat import OpenAIChat
from tinyllm.util import os_util
from tinyllm.functions.llms.openai.helpers import get_system_message, get_user_message

openai.api_key = os.getenv("OPENAI_API_KEY")
loop = asyncio.get_event_loop()

system_msg = get_system_message("""
You are a Python superhuman developer and expert of the tinyllm library code, documentation and logic. 
""")
system_msg = get_user_message("""
You help developers understand the tinyllm library and code with tinyllm. Guidelines:
- When creating a new function, you always include the Function, input_validator and output_validator.
""")
openai_chat = OpenAIChat(name='TinyLLM Agent',
                         llm_name='gpt-3.5-turbo-16k',
                         temperature=0,
                         prompt_template=[
                             system_msg,
                         ],
                         n=1,
                         verbose=True)

default_cache_path = os_util.joinPaths([os_util.getUserHomePath(), 'tinyllm_cache', 'cache.json'])
local_file_cache = LocalFilesCache(cache_path=default_cache_path,
                                   source_dir='/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/')
local_file_cache.refresh_cache()

def tinyllm_chat(message, history):
    # Always check if any new files have been added to the library
    local_file_cache.refresh_cache()
    files = local_file_cache.get_similar_files(message, n=3)

    # Always include the Function class
    function_class = local_file_cache.get_file_content('/Users/othmanezoheir/PycharmProjects/openagents/tiny-llm/tinyllm/functions/function.py')
    files.append(function_class)

    # Create the context
    context = "\n".join(files)
    message = context + "\n----------\n"+"User question: " + message + "\n"

    # Get the response from the AI agent
    chat_response = loop.run_until_complete(openai_chat(message=message))
    return chat_response['response']




CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 80vh !important; }
#component-0 { height: 80%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""
demo = gr.ChatInterface(tinyllm_chat,
                        chatbot=gr.Chatbot(height=500),
                        textbox=gr.Textbox(placeholder="Ask any question", container=True, scale=15, lines=6),
                        title="Tinyllm AI Agent",
                        examples=["How do I build a chain with tinyllm?"],
                        cache_examples=False,
                        retry_btn="Try Again",
                        undo_btn=None,
                        clear_btn="Clear",
                        )
demo.launch(debug=True,
            inbrowser=True)
