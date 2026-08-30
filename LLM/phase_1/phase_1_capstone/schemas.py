from pydantic import BaseModel, validate_call
from typing import List

@validate_call
class DocumentSummary(BaseModel):
    """
    Output - Endpoint 1: LLM Output Schema for document summarization
    """
    title: str 
    summary: str
    key_points: List[str]
    keywords: List[str]

@validate_call
class SummarizeRequest(BaseModel):
    """
    Input - Endpoint 1: Document to Summarize
    """
    text: str # Text for the LLM
@validate_call
class QueryRequest(BaseModel):
    """
    Input - Endpoint 2: Input from User
    """
    context_text: str # Context from Files
    question: str # Question for the LLM

