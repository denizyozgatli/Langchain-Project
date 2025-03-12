from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represent the state pf a graph
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search or not
        documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]