# RAG --> Retrieval Augmented Generation

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(
    documents = documents,
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
# Gemini'nin verdiğimiz documenta bağlı kalması için .bins(k=1) yaptık

llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",
                               temperature=0.1)

message = """
Answer this question using the provided context only.
{question}

Context: 

{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {"context" : retriever, "question" : RunnablePassthrough()} | prompt | llm

if __name__ == "__main__":
    response = chain.invoke("tell me about cat")
    print(response.content)