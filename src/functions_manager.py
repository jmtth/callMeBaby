import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
)

ParameterType = Literal["number", "string", "boolean"]


TYPE_MAPPING: dict[ParameterType, type] = {
    "number": float,
    "string": str,
    "boolean": bool,
}


class Parameter(BaseModel):
    """Class representing a function parameter."""
    type: ParameterType = Field(..., description="Type of the parameter")


class FunctionSchema(BaseModel):
    """Class representing a function schema."""
    name: str = Field(..., description="Name of the function")
    description: str = Field("", description="Description of the function")
    parameters: dict[str, Parameter] = Field(
        default_factory=dict,
        description="Parameters",
    )


class FunctionsDefinition:
    """Class to load and query
    function definitions from a JSON file.
    Args:
        path_to_json(str): JSON file path

    Returns:
        list[str]: list of function names
        FunctionSchema: function definition by name
        str: function description by name
        dict: function parameters by name
        int: function parameters count by name

    Raise:
        ValueError: if function name not found
        ValueError: if JSON file is missing or malformed
        ValueError: if unexpected error occurs
        ValueError: if function definition is invalid
    """

    def __init__(self, functions: list[FunctionSchema]):
        names = [function.name for function in functions]
        if len(names) != len(set(names)):
            raise ValueError("Function names must be unique")
        self.functions = functions
        self._functions_by_name = {
            function.name: function for function in functions
        }

    @classmethod
    def from_json(cls, path_to_json: str) -> "FunctionsDefinition":
        """Load function definitions from a JSON file
        and return an instance of FunctionsDefinition.
        """
        try:
            raw_text = Path(path_to_json).read_text(encoding="utf-8")
            data = json.loads(raw_text)
            if not isinstance(data, list):
                raise ValueError("Function definitions must be a JSON list")
            functions = [FunctionSchema(**func) for func in data]
            return cls(functions)
        except FileNotFoundError as exc:
            raise ValueError(f"File not found: {path_to_json}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file: {path_to_json}") from exc
        except (TypeError, ValidationError) as exc:
            raise ValueError(f"Invalid function definition in: {exc}") from exc

    def list_functions_name(self) -> list[str]:
        """Return a list of function names in the order they were defined."""
        return [func.name for func in self.functions]

    def get_function_by_name(self, name: str) -> FunctionSchema:
        """Return the function definition for the given name."""
        try:
            return self._functions_by_name[name]
        except KeyError as exc:
            raise ValueError(f"Function with name '{name}' not found") from exc

    def get_function_description_by_name(self, name: str) -> str:
        """Return the function description for the given name."""
        func = self.get_function_by_name(name)
        return func.description

    def get_function_parameters_by_name(
        self,
        name: str,
    ) -> dict[str, Parameter]:
        """Return the function parameters for the given name."""
        func = self.get_function_by_name(name)
        return func.parameters

    def get_nb_parameters(self, name: str) -> int:
        """Return the number of parameters for the function
        with the given name.
        """
        params = self.get_function_parameters_by_name(name)
        return len(params)

    def get_functions_prompt(self) -> str:
        """Return a string representation of all functions
        in a format suitable for prompting.
        """
        prompt = "Here are the available functions:\n\n"
        for func in self.functions:
            prompt += f" - Function Name: {func.name}\n"
            prompt += f"   Description: {func.description}\n"
            prompt += "   Parameters:\n"
            for param_name, param in func.parameters.items():
                prompt += f"    - {param_name} (type: {param.type})\n"
            prompt += "\n"
        return prompt

    def get_output_function_model(self, name: str) -> type[BaseModel]:
        """Return a Pydantic model for the output of the function
        with the given name.
        The model will have the following fields:
        - prompt: str
        - name: Literal[func.name]
        - parameters: ParamsModel
        """
        func = self.get_function_by_name(name)
        params_fields: dict[str, Any] = {
            param_name: (TYPE_MAPPING[param.type], ...)
            for param_name, param in func.parameters.items()
        }
        ParamsModel: type[BaseModel] = create_model(
            f"{func.name}_params",
            __config__=ConfigDict(extra="forbid"),
            **params_fields,
        )
        OutputSchema: type[BaseModel] = create_model(
            f"{func.name}_output",
            prompt=(str, ...),
            name=(Literal[func.name], func.name),
            parameters=(ParamsModel, ...),
            __config__=ConfigDict(extra="forbid"),
        )
        return OutputSchema
