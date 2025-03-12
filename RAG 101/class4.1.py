from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",
                               temperature=0.1)

#1.Adım: İnternetten veri çekilecek
#2.Adım: Veriyi text splitterlar ile bölünecek, GoogleGenAIEmbeddings kullanarak vektörize edip database kaydedeceğiz
#3.Adım: Kaydettikten sonra retreiver oluşturacağız. Sonra LLM'e rag yaptığımızı söyleyip bilgi isteyeceğiz

#1.Adım
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

#2. 3. Adımlar
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
#GoogleGenerativeAIEmbeddings() yerine Hugginface'den farklı embeddings modelleri ile çalışabilirsin

retriever = vectorstore.as_retriever()
#rag prompt
promt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
    | promt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    for chunk in rag_chain.stream("What is task decomposition?"):
        print(chunk, end="", flush=True)

#Soru1: What is maximum inner product search?
#Soru2: What is task decomposition?