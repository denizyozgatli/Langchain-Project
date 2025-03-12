from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",
                               temperature=0.1)

messages = [
    SystemMessage(content="Translate the following from English to Spanish"),
    HumanMessage(content="Hi!")
]

if __name__ == "__main__":
    response = model.invoke(messages)
    print(response.content)
