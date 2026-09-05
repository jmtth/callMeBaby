from pydantic import BaseModel, Field


class PromptSchema(BaseModel):
    """Describe a single function call request."""
    prompt: str = Field(..., description="User request to process")
