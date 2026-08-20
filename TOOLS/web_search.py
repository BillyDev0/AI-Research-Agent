import os
from tavily import TavilyClient

client=TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY"),
    )

def web_search(query):

    response=client.search(
        query=query,
        max_results=5,
        search_depth="basic"
    )

    return response["results"]
