
#installation

```bash
uv init <projet_name>
#or if the directory exist
uv init --bare

unzip data.zip
unzip llm_sdk.zip

#add workspace in root pyproject.toml
[tool.uv.workspace]
members = ["llm_sdk"]

# add dependecies
uv add --editable ./llm_sdk

# add mandatory package 
uv add numpy json

# synchronise 
uv sync

```