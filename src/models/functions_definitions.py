from pydantic import BaseModel, Field
from typing import Literal

ParameterType = Literal["number", "integer", "string", "boolean"]


TYPE_MAPPING: dict[ParameterType, type] = {
    "number": float,
    "integer": int,
    "string": str,
    "boolean": bool,
}


class Parameter(BaseModel):
    """Describe one function parameter."""
    type: ParameterType = Field(..., description="Type of the parameter")


class FunctionSchema(BaseModel):
    """Describe a callable function and its parameters."""
    name: str = Field(..., description="Name of the function")
    description: str = Field("", description="Description of the function")
    parameters: dict[str, Parameter] = Field(
        default_factory=dict,
        description="Parameters",
    )
