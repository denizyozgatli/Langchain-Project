from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",
                               temperature=0)

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()