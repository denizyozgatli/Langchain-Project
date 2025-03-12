from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",
                               temperature=0.1)

system_prompt = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), ("user", "{text}")
    ]
)

parser = StrOutputParser()
chain = prompt_template | model | parser

app = FastAPI(
    title="Translator App!",
    version="1.0.0",
    description="Translation Chat Bot",
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)

# LANGSERVE
# http://localhost:8000/chain/playground/