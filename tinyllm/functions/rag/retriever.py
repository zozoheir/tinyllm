import inspect
from textwrap import dedent

import rumorz_backend.graph_db.kg_model as kg_model
from rumorz_llms.util.models import minilm_embedding_function

from rumorz_llms.util.rumorz_graph import RumorzGraph
from tinyllm.function import Function
from tinyllm.functions.llm.util import get_assistant_message, get_user_message
from tinyllm.functions.rag.multi_source_context_builder import MultiSourceDocsContextBuilder
from tinyllm.functions.llms.openai.openai_prompt_template import OpenAIPromptTemplate
from tinyllm.functions.rag.pg_vector_store import VectorStore

GRAPH_SCHEMA = inspect.getsource(kg_model)
vector_store = VectorStore(embedding_function=minilm_embedding_function)
rumorz_graph = RumorzGraph()

########################## PROMPT TEMPLATE ##########################
INPUT_EXAMPLE = """
------------------------
[document]
- text:...
- url: https://www.xrp.com
- timestamp: 1 week ago, 2023-11-01 22:55:30 
- source: XRP.com
[document]
- text:...
- url: https://www.markets.com
- timestamp: 2 weeks ago, 2023-10-24 22:55:30 
- source: Markets
- url: https://www.markets.com

------------------------
<ANSWER FORMAT>
<USER QUESTION>:
What is the outlook on XRP
"""
OUTPUT_EXAMPLE = dedent("""
A **seasoned crypto strategist** forecasts an imminent surge for XRP, with expectations set on an ascent to the **$0.62-$0.63** 
range, contingent upon the asset breaching the pivotal resistance level at **$0.548**. [XRP.com](https://www.xrp.com)

In concurrence, **Valhil Capital's analytical assessment** posits that XRP's current market position is undervalued, with 
its valuation potential outstripping that of Bitcoin. The firm’s extensive analysis, utilizing six distinct pricing 
frameworks, places XRP's intrinsic value between **$9.81 to $513,000**. [Markets](https://www.markets.com)
""")
examples = [
    get_user_message(INPUT_EXAMPLE),
    get_assistant_message(OUTPUT_EXAMPLE),
]

kg_qa_chain_prompt_template = OpenAIPromptTemplate(
    name="KG QA Chain Prompt Template",
    system_role=dedent(f"""
ROLE:
You are a cutting-edge AI agent Financial Markets and Trading and Investing expert. 

TASK:
- You will be given knowledge from the Rumorz KNOWLEDGE GRAPH. Use the KNOWLEDGE GRAPH answer the user question in the most accurate way possible. 
- If the KNOWLEDGE GRAPH to answer the question, just say "I don't have enough information to answer this question".
- Each knowledge has a "source" and "url" for quotes
"""),
    is_traced=False,
    messages=[],  # examples
)
ANSWER_FORMAT_PROMPT = f"""
ANSWER FORMAT
- Answer in a conversational way, talk like Financial markets expert at top hedge fund
- Your answers should be short and well organized in Markdown format when appropriate
- Use paragraphs to separate different ideas
- Answer in Markdown format
- DO NOT UNDER ANY CIRCUMSTANCES, NEVER use "According to/Based on the news/source" or similar phrasing type
- ALWAYS use Markdown hyperlinks using the document's source and url to cite your source. For example:
```
- source: Markets
- url: https://www.markets.com
```
[Markets](https://www.markets.com)"

NOW
- Use the Rumorz Knowledge Graph to answer the following question
"""


################ THE CHAIN #################


class Retriever(Function):
    def __init__(self,
                 is_traced=True,
                 context_builder=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.context_builder = context_builder

    async def run(self, **kwargs):
        retrieved_docs = await self.search(**kwargs)
        context = await self.build_context(retrieved_docs=retrieved_docs,
                                           **kwargs)
        return {"context": context}

    async def search(self, **kwargs):
        docs = []
        return docs

    async def build_context(self, **kwargs):
        final_context = self.context_builder.get_context(
            docs=kwargs['retrieved_docs'],
        )
        return final_context


context_builder = MultiSourceDocsContextBuilder(
    start_string="<SOURCE DOCS START>",
    end_string="<SOURCE DOCS END>",
    available_token_size=None
)
