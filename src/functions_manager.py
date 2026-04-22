import json
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List


class Parameter(BaseModel):
    """Class representing a function parameter."""
    type: str = Field(..., description="Type of the parameter")


class FunctionSchema(BaseModel):
    """Class representing a function schema."""
    name: str = Field(..., description="Name of the function")
    description: str = Field("", description="Description of the function")
    parameters: dict[str, Parameter] = Field({}, description="Parameters")


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

    def __init__(self, functions: List[FunctionSchema]):
        self.functions = functions

    @classmethod
    def from_json(cls, path_to_json: str) -> "FunctionsDefinition":
        """Load function definitions from a JSON file 
        and return an instance of FunctionsDefinition.
        """
        try:
            with open(path_to_json, "r") as f:
                data = json.load(f)
            functions = [FunctionSchema(**func) for func in data]
            return cls(functions)
        except FileNotFoundError as exc:
            raise ValueError(f"File not found: {path_to_json}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file: {path_to_json}") from exc
        except ValidationError as exc:
            raise ValueError(f"Invalid function definition in: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Unexpected error : {exc}") from exc

    def list_functions_name(self) -> list[str]:
        return [func.name for func in self.functions]

    def get_function_by_name(self, name: str) -> FunctionSchema:
        for func in self.functions:
            if func.name == name:
                return func
        raise ValueError(f"Function with name '{name}' not found")

    def get_function_description_by_name(self, name: str) -> str:
        func = self.get_function_by_name(name)
        return func.description

    def get_function_parameters_by_name(self, name: str) -> Dict:
        func = self.get_function_by_name(name)
        return func.parameters

    def get_nb_parameters(self, name: str) -> int:
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

def main():
    functions_def = FunctionsDefinition.from_json(
        "data/input/functions_definition.json")
    print(functions_def.list_functions_name())
    name = "fn_substitute_string_with_regex"
    print(functions_def.get_function_by_name(name))
    print(functions_def.get_function_description_by_name(name))
    print(functions_def.get_function_parameters_by_name(name))
    print(functions_def.get_nb_parameters(name))


if __name__ == "__main__":
    main()
