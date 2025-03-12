from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ÖNEMLİ: langchain-chroma yüklenirken hata alırsan visual studio installerdan Win11 SDK indirerek sorunu çözebilirsin

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
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") #Google GENAI Embeddings kullanırken model belirtmek gerekli
)

"""
#1
if __name__ == "__main__":
    #print(vectorstore.similarity_search("dog"))
    print(vectorstore.similarity_search_with_score("dog")) #..._with_score --> verdiği dönüşü skorlayarak gösterir
"""

#2
if __name__ == "__main__":
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector = embedding.embed_query("dog")
    print(vectorstore.similarity_search_by_vector(vector))
