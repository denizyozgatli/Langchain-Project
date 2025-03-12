from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",
                               temperature=0.1)

search = TavilySearchResults(max_results=2)

tools = [search]

agent_executor = create_react_agent(model, tools)

if __name__ == "__main__":
    response = agent_executor.invoke(
        {"messages" : [HumanMessage(content="What is the weather in Ankara now?")]},
    )
    for r in response["messages"]:
        print(r.content)
        