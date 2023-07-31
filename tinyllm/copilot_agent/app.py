import asyncio

import gradio as gr
import os
import openai
from tinyllm.functions.llms.openai.openai_chat_agent import OpenAIChatAgent
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate


openai.api_key = os.getenv("OPENAI_API_KEY")

loop = asyncio.get_event_loop()

prompt_template = OpenAIPromptTemplate(
    name="TinyLLM Agent Prompt Template",
    system_role="""
    # ROLE
    You are the TinyLLM agent, a world-class Python developer and copilot for users of a new library, tinyllm. 
    
    # KNOWLEDGE
    It is July 2023;
    
    # FUNCTIONS
    You have access the tinyllm library code, documentation and logic through a function. Use the function to answer the user's questions.
    You help users understand tinyllm and build chains, llm workflows and all types of tinyllm functions.
    
    # TASK:
    Answer the user's questions about tinyllm. 
    You must use namespace functions to obtain the information and documentation required to answer any questions.
    
    """,
    messages=[
    ],
)

functions = [
        {
            "name": "tinyllm_agent",
            "description": "This is your default tool. This function gives you access to tinyllm documentation, code, classes, examples on using the tinyllm library. Use this function when the user asks a question about the tinyllm library or writing functions, chains or code with tinyllm",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The user input to the tinyllm agent.",
                    },
                },
                "required": ["input"],
            },
        }
    ]

openai_agent = OpenAIChatAgent(
    name="TinyLLM Agent",
    llm_name="gpt-3.5-turbo",
    openai_functions=functions,
    temperature=0,
    prompt_template=prompt_template,
    n=1,
    verbose=True,
)



def tinyllm_chat(message, history):
    # Get the response from the AI agent
    chat_response = loop.run_until_complete(openai_agent(message=message))
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
