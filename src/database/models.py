from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class ApplicationStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    MATCHED = "MATCHED"
    SKIPPED = "SKIPPED"
    RESUME_GENERATED = "RESUME_GENERATED"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPLIED = "APPLIED"
    APPLIED_MANUALLY = "APPLIED_MANUALLY"
    FAILED = "FAILED"

class JobListing(BaseModel):
    id: Optional[str] = None
    source: str  # linkedin, greenhouse, lever, ashby, custom
    url: str
    title: str
    company: str
    location: Optional[str] = None
    description: Optional[str] = None
    is_remote: bool = False
    fit_score: Optional[int] = None
    fit_reasoning: Optional[str] = None
    tailored_resume_path: Optional[str] = None
    application_kit_path: Optional[str] = None
    status: ApplicationStatus = ApplicationStatus.DISCOVERED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    applied_at: Optional[datetime] = None
    error_message: Optional[str] = None
