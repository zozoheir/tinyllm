import asyncio
import random
import gradio as gr
import os
import openai
import tinyllm
from tinyllm.ai_util import find_related_files, get_tinyllm_embeddings
from tinyllm.functions.llms.openai.helpers import get_user_message, get_system_message
from tinyllm.functions.llms.openai.openai_chat import OpenAIChat
from tinyllm.prompt_util import listDir

openai.api_key = os.getenv("OPENAI_API_KEY")
loop = asyncio.get_event_loop()

tinyllm_embeddings_path = os.path.join(tinyllm.__path__[0], 'tinyllm_embeddings.pickle')
library_files = listDir(tinyllm.__path__[0], recursive=True, format='py')
library_files = [file for file in library_files if not file.endswith('tinyllm/ai.py')]

tinyllm_embeddings = get_tinyllm_embeddings(library_files, tinyllm_embeddings_path)

related_files = find_related_files('I need to write a new Chain',
                                   tinyllm_embeddings)

# Always include the Function class
function_file = [file for file in tinyllm_embeddings if file['file_path'].endswith('function.py')][0]
related_files.append(function_file)

context = "\n".join(file['content'] for file in related_files)

system_msg = get_system_message("""
You are a Python superhuman developer and expert of the tinyllm library. You help developers understand and code with 
tinyllm
""")

context = get_user_message(f"""
Below are supporting files from the tinyllm library to help answer the question

{context}
""")

openai_chat = OpenAIChat(name='OpenAI Chat model',
                         llm_name='gpt-3.5-turbo',
                         temperature=0,
                         prompt_template=[
                             system_msg,
                             context
                         ],
                         n=1,
                         verbose=True)


def random_response(message, history):
    chat_response = loop.run_until_complete(openai_chat(message=message))
    return chat_response['response']
CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 80vh !important; }
#component-0 { height: 80%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

demo = gr.ChatInterface(random_response,
                        css=CSS,
                        chatbot=gr.Chatbot(height=700),
                        textbox=gr.Textbox(placeholder="Ask any question", container=False, scale=7),
                        title="Tinyllm Programmer",
                        examples=["How do I build a chain with tinyllm?"],
                        cache_examples=False,
                        retry_btn="Try Again",
                        undo_btn=None,
                        clear_btn="Clear",
                        )

demo.launch(debug=True,
            inbrowser=True)