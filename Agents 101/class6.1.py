from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",
                               temperature=0.1)

search = TavilySearchResults(max_results=2)

tools = [search]

model_with_tools = model.bind_tools(tools)

if __name__ == "__main__":
    response = model_with_tools.invoke([HumanMessage(content="What is the current weather in Ankara now?")])
    print(response.content)
    print(response.tool_calls)