from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
#from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",
                               temperature=0.1)

memory = SqliteSaver.from_conn_string(":memory:")
search = TavilySearchResults(max_results=2)

prompt = hub.pull("hwchase17/react")

tools = [search]

agent = create_react_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, checkpoint=memory)

config = {"configurable": {"thread_id": "abc123"}}

if __name__ == "__main__":
    chat_history = []
    while True:
        user_input = input(">")
        response = []
        for chunk in agent_executor.stream(
                {"input": user_input,
                "chat_history" : "\n".join(chat_history)
                 },
                config=config
        ):
            if 'text' in chunk:
                print(chunk['text'], end="")
                response.append(chunk['text'])

        chat_history.append(f"AI: {''.join(response)}")