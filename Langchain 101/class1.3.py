from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",
                               temperature=0.1)

messages = [
    SystemMessage(content="Translate the following from English to Spanish"),
    HumanMessage(content="Hi!")
]

parser = StrOutputParser()

chain = model | parser
#response = model.invoke(messages)

if __name__ == "__main__":
    print(chain.invoke(messages))
    #print(parser.invoke(response))
