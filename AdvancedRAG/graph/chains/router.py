from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

class RouterQuery(BaseModel):
    """
    Route a user query to the most relevant datasource
    """

    datasource : Literal["vectorstore","websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",
                               temperature=0)
structured_llm_router = llm.with_structured_output(RouterQuery)

system_prompt = """
You are an expert at routing a user question to a vectorstore or a web search.
The vectorstore contains documents related to agents, prompt engineering and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

question_router = route_prompt | structured_llm_router