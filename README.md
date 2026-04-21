
# installation

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
uv add numpy json mypy flake8

# synchronise 
uv sync

```

# Choix des Librairies
- Numpy
  - Gain de performance sur la gestion des tableaux
    > - cela est tres pertinent pour manipuler les logits
    > - plusieurs milliers de tokens ici 151936
    > - logits.index(max(logits)) -> int(np.argmax(logits))
  - Gestion simplifié des mask
    > - mask = np.full_like(logits_np, float("-inf"))
  - Fancy indexing
    > - Evite boucle for avec de nombreuse comparaison
    > - indices = list(allowed_ids)
    > - mask[indices] = logits_np[indices]


# Principe
Le processus de génération d'un LLM fonctionne toujours en convertissant le texte 
en IDs numériques pour que le réseau de neurones puisse les traiter. 
- Le Vocabulaire : 
> Le fichier JSON renvoyé par `get_path_to_vocabulary_json()`
> est un immense dictionnaire qui fait la correspondance entre un ID
> et sa représentation textuelle (ex: fn_add) : le `vocab`.
- L'Entrée : 
> Le prompt est transformé en une liste d'IDs via `model.encode()` 
> avant d'être envoyé au modèle.
- La Sortie : 
> Le modèle ne "voit" pas les mots. 
> Il calcule des probabilités (logits) pour chaque ID présent dans son vocabulaire.
- Les contraintes : 
> En utilisant mes allowed_tokens, 
> je force le modèle à choisir uniquement parmi les IDs qui correspondent 
> à des séquences de caractères valides pour mon schéma JSON.