from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05")

"""
if __name__ == "__main__":
    #1 message = HumanMessage(content="Hello, my name is Deniz")
    #2 message = HumanMessage(content="What is my name?")
    response = model.invoke([message])
    print(response.content)
"""

if __name__ == "__main__":
    messages = [
        HumanMessage(content="Hello, my name is Deniz"),
        AIMessage(content="Hello Deniz, how can i help you today?"),
        HumanMessage(content="What is my name?"),
    ]
    response = model.invoke(messages)
    print(response.content)