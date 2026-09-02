*This project has been created as part of the 42 curriculum by jhervoch.*

# Call Me Maybe

## Description

Call Me Maybe translates natural-language requests into typed JSON function
calls. It uses Qwen3-0.6B through the provided `llm_sdk` and applies constrained
decoding so the model can only generate tokens compatible with the functions
and parameter schemas loaded from `functions_definition.json`.

The program does not execute the selected function. It produces objects with
exactly these fields:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

## Instructions

Install the dependencies:

```bash
make install
```

Run with the default files:

```bash
make run
```

Or provide another function set, prompt file, and output path:

```bash
uv run python -m src \
  --functions_definition path/to/functions_definition.json \
  --input path/to/prompts.json \
  --output path/to/function_calling_results.json
```

By default, the CLI reads:

- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`

and creates `data/output/function_calling_results.json`.

Useful development commands:

```bash
make test
make coverage
make lint
make debug
make clean
```

## Generation Visualizer

Open the interactive Textual dashboard with:

```bash
make visualize
```

Custom input and output paths remain available:

```bash
uv run python -m src --visualize \
  --functions_definition path/to/functions_definition.json \
  --input path/to/prompts.json \
  --output path/to/function_calling_results.json
```

The dashboard follows each response in real time. It displays the current FSM
state, the JSON under construction, the generated-token count, deterministic
structure insertions, and the twelve highest-logit tokens that remain valid
under the active constraint. Generation runs in a background worker so the
interface remains responsive. Press `q` to quit or `c` to clear the event log.

## Algorithm

The generation pipeline works as follows:

1. `FunctionsDefinition` parses and validates every function using Pydantic.
2. `build_prompt` gives the model the names, descriptions, parameter names,
   and parameter types from the selected definition file.
3. The prompt is encoded into token IDs with the public SDK API.
4. `TokenVocabulary` indexes decoded token text and reusable token classes.
5. `JSONStateMachine` tracks the position in the required output schema.
6. Fixed JSON fragments, such as keys and punctuation, are emitted directly.
7. At dynamic states, the FSM computes the token IDs that keep the partial
   output compatible with the schema.
8. Only allowed logits are compared, and the highest remaining logit selects
   the next token. If only one token is structurally possible, it is emitted
   directly without an unnecessary model inference.
9. `GenerationBuffer` owns the generated IDs, model context, and exact response
   token budget, avoiding divergent copies of the generated sequence.
10. Once a complete function name has been selected by the LLM, the FSM loads
   that function's parameters and constrains each parameter name and value to
   its declared type. If the prompt lacks enough number or boolean literals,
   generation stops immediately with an explicit error.
11. The completed JSON is parsed and validated again with a dynamically created
   Pydantic output model.

Function selection is performed by the LLM logits. The FSM does not use
keywords or hardcoded knowledge of the demonstration function names; it only
restricts the output to functions present in the supplied definition file.

## Design Decisions

- Function definitions are loaded dynamically so peer-review files can contain
  different names and schemas.
- A state machine separates structural JSON generation from semantic choices.
- Fixed sequences bypass model inference because their tokens contain no choice.
- Function names are generated from prefixes that remain valid for at least one
  supplied function. A boundary token explicitly distinguishes a short name
  from a longer function sharing the same prefix.
- Parameter names follow the order declared in the input schema.
- Number generation accepts only valid numeric fragments and forces a delimiter
  when the expected precision is complete.
- String values have a bounded length. Unsafe unescaped JSON fragments and
  fragments that would create a repetition are rejected before insertion.
- Boolean literals support both single-token and multi-token tokenizers and
  advance the FSM immediately after `true` or `false` is complete.
- Functions with no parameters use a dedicated state that emits an empty
  `parameters` object and closes the outer JSON object correctly.
- Pydantic validates both input definitions and generated output types.

## Performance Analysis

The main cost is `get_logits_from_input_ids`, because it runs a model forward
pass for each token that requires a semantic choice. Fixed JSON sequences and
single-token structural choices are emitted without a logits call. This reduces
work for common prefixes, parameter names, punctuation, and closing sequences.

The implementation compares logits only at allowed token positions before
`argmax` selects the best valid token. Model loading and the decoded vocabulary
cache are reused for every prompt in the input file.

Actual runtime depends on CPU, CUDA, or MPS performance and should be measured
with the complete evaluation input rather than a single prompt.

## Challenges Faced

- Tokenizer entries may contain leading-space markers and may split function
  names at unexpected positions.
- JSON strings require escaped prompt content and carefully controlled quotes.
- Numbers need a clear stopping condition; continuing to allow digits after a
  complete value can exhaust the response limit and leave invalid JSON.
- Functions with no parameters require a different closing sequence from
  functions with one or more parameters.
- Small language models may repeat string fragments, so the decoder bounds
  string length and detects repeating patterns.
- Function files used during review may differ from the examples, so the
  decoder must derive every name and parameter constraint from input data.

## Testing Strategy

The test suite covers:

- malformed, empty, and missing JSON input files;
- both supported vocabulary mapping shapes;
- function-definition parsing and dynamic Pydantic models;
- fixed and dynamic FSM transitions;
- escaped quotes in the original prompt;
- string, number, and empty-parameter generation;
- numeric fragments, precision, and termination tokens;
- proactive repetition prevention without token re-encoding;
- boolean completion and function names with shared prefixes;
- exact, atomic response-token budget enforcement;
- a state-step guard and early rejection of missing typed prompt values;
- model-loading error handling without loading a real model in unit tests;
- CLI defaults and response generation branches.

Run all tests with:

```bash
make test
```

Run the mandatory style and type checks with:

```bash
make lint
```

## Example Usage

Input function definition:

```json
{
  "name": "fn_greet",
  "description": "Generate a greeting for a person.",
  "parameters": {
    "name": {
      "type": "string"
    }
  }
}
```

Input prompt:

```json
{
  "prompt": "Greet Ada"
}
```

Expected output shape:

```json
{
  "prompt": "Greet Ada",
  "name": "fn_greet",
  "parameters": {
    "name": "Ada"
  }
}
```

## Resources

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- The project subject and the supplied `llm_sdk` package.

AI was used as a review and learning aid to explain constrained-decoding state
transitions, identify schema and token-stopping bugs, and suggest unit-test cases.
AI was used for regex and vizualisation dashboard.
Every proposed change was inspected and validated locally with pytest, flake8,
and mypy.

## Technical learnings

### Create project

```bash
uv init <project_name>
# or if the directory exists
uv init --bare

unzip data.zip
unzip llm_sdk.zip

# add workspace in root pyproject.toml
[tool.uv.workspace]
members = ["llm_sdk"]

# add dependencies
uv add --editable ./llm_sdk

# add mandatory packages
uv add numpy json mypy flake8

# synchronize
uv sync
```

### Code quality
- **PEP 8**
  Very useful for clean code, easy to read, scalable, and maintainable
  > Syntax rules like:
  > - space indentation
  > - indentation for '}', ']', ')' on multiple lines
  > - maximum line length
  > - newline rules:
  >   - before or after an operator
  > - import rules
  > - etc...

  https://www.flake8rules.com/
  https://peps.python.org/pep-0008/

- **PEP 257**
  Code documentation rules for adding docstrings to classes and functions
  > - Every module and public method must have a docstring
  > - Docstrings must start with `"""` and end with `"""`.
  > - Sentences explain what the functions do
  > - Multi-line docstrings start with `"""` followed on the same line by the function description and must end with a newline and `"""`

  It is possible to improve this standard by adding args, returns, and errors.
  > For this, I chose the Google style.

  Example:
  ```python
  """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
  ```

  https://peps.python.org/pep-0257/

### Library choice
- NumPy
  - Performance gain in array handling
    > - this is particularly relevant for manipulating logits
    > - there are several thousand tokens here, 151936
    > - logits.index(max(logits)) -> int(np.argmax(logits))
  - Simplified mask management
    > - mask = np.full_like(logits_np, float("-inf"))
  - Fancy indexing
    > - Avoids for loops with many comparisons
    > - indices = list(allowed_ids)
    > - mask[indices] = logits_np[indices]

- Pydantic
  - Validation of input and output data
  - Dynamic generation of Pydantic models for functions and parameters
  - Handling of complex types (lists, dictionaries, etc.)
  - Management of default values and validation constraints
  - Automatic generation of model documentation

-Protocol
  - Protocols are a way to define interfaces in Python
  - They allow for structural subtyping, meaning that a class can be considered
    a subtype of a protocol if it has the required methods and attributes,
    regardless of its inheritance hierarchy.
  - This is useful for defining expected behavior without enforcing a strict
    inheritance structure.

-Annotations
  - Type annotations provide a way to specify the expected types of variables,
    function parameters, and return values.
  - They help with static type checking, code readability, and documentation.
  - Annotations can be used with tools like mypy to catch type errors before
    runtime.


