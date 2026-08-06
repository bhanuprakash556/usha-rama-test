# pip install -U langchain langchain-openrouter langchain-community duckduckgo-search
# pip install -U ddgs

import os

from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun

os.environ["OPENROUTER_API_KEY"] = (
    " "
)

agent = create_agent(
    model="openrouter:google/gemma-4-26b-a4b-it:free",
    tools=[DuckDuckGoSearchRun()],
    system_prompt="You are a helpful assistant",
)


def get_response(prompt: str) -> str:
    """Take a prompt and return the agent response as a string."""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(get_response("What's the weather in Vijayawada"))
