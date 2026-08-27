from pydantic import BaseModel
from typing import List

class DocumentSummary(BaseModel):
    """
    Output - Endpoint 1: LLM Output Schema for document summarization
    """
    title: str 
    summary: str
    key_points: List[str]
    keywords: List[str]

class SummarizeRequest(BaseModel):
    """
    Input - Endpoint 1: Document to Summarize
    """
    text: str # Text for the LLM

class QueryRequest(BaseModel):
    """
    Input - Endpoint 2: Input from User
    """
    context_text: str # Context from Files
    question: str # Question for the LLM

