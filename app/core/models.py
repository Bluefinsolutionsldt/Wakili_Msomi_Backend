"""
Data models for Sheria Kiganjani
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Metadata for a document"""
    filename: str
    file_type: str
    size: int
    upload_date: datetime = Field(default_factory=datetime.now)
    language: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class DocumentAnalysisResponse(BaseModel):
    """Response model for document analysis"""
    summary: str
    key_points: List[str]
    recommendations: List[str]
    metadata: DocumentMetadata
    confidence_score: float = Field(ge=0.0, le=1.0)
    processed_at: datetime = Field(default_factory=datetime.now)

class QueryRequest(BaseModel):
    """Request model for legal queries"""
    query: str
    language: str = "sw"
    conversation_id: str
    document_id: Optional[str] = None
    is_offline: bool = False
    metadata: Dict = Field(default_factory=dict)

class QueryResponse(BaseModel):
    """Response model for legal queries"""
    response: str
    language: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    conversation_id: Optional[str] = None
    document_references: List[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)
