import json


class FunctionsDefinition:
    """Class to load and query
    function definitions from a JSON file.
    Args:
        path_to_json(str): JSON file path

    Returns:
        list[str]: list of function names
        dict: function definition by name or id
        str: function description by name
        dict: function parameters by name
        int: function parameters count by name

    Raise:
        ValueError: if function name or id not found

    """
    functions: list[dict]

    def __init__(self, path_to_json: str):
        try:
            with open(path_to_json) as f:
                data = json.load(f)
        except Exception as exc:
            raise ValueError(f"Error loading JSON file: {exc}") from exc
        self.functions = []
        for funcs in data:
            self.functions.append(funcs)

    def list_functions_name(self) -> list[str]:
        return [func['name'] for func in self.functions]

    def get_function_by_name(self, name: str) -> dict:
        for func in self.functions:
            if func['name'] == name:
                return func
        raise ValueError(f"Function with name '{name}' not found")

    def get_function_by_id(self, id: str) -> dict:
        for func in self.functions:
            if func['id'] == id:
                return func
        raise ValueError(f"Function with id '{id}' not found")

    def get_function_description_by_name(self, name: str) -> str:
        func = self.get_function_by_name(name)
        return func.get('description', "")

    def get_function_parameters_by_name(self, name: str) -> dict:
        func = self.get_function_by_name(name)
        return func.get('parameters', {})

    def get_function_parameters_count_by_name(self, name: str) -> int:
        params = self.get_function_parameters_by_name(name)
        return len(params)


def main():
    functions_def = FunctionsDefinition("data/input/functions_definition.json")
    print(functions_def.list_functions_name())
    name = "fn_substitute_string_with_regex"
    print(functions_def.get_function_by_name(name))
    print(functions_def.get_function_description_by_name(name))
    print(functions_def.get_function_parameters_by_name(name))
    print(functions_def.get_function_parameters_count_by_name(name))


if __name__ == "__main__":
    main()
