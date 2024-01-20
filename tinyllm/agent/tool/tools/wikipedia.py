import requests

from tinyllm.agent.tool.tool import Tool


def get_wikipedia_summary(page_title):
    """ Fetch the summary of a Wikipedia page given its title. """
    URL = "https://en.wikipedia.org/api/rest_v1/page/summary/" + page_title
    try:
        response = requests.get(URL)
        data = response.json()
        return data.get('extract', 'No summary available.')
    except requests.RequestException as e:
        return str(e)


def get_wikipedia_summary_tool():
    return Tool(
        name="get_wikipedia_summary",
        description="""
        Use this tool to get a Wikipedia summary. 
        """,
        python_lambda=get_wikipedia_summary,
        parameters={
            "type": "object",
            "required": ["page_title"],
            "properties": {
                "page_title": {
                    "type": "string",
                    "description": "The wikipedia page_title (no spaces, only '_'). Example: Elon_Musk, Python_(programming_language)",
                },
            }
        }
    )
