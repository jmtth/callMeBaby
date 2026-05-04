
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

# Qualite du code 
- **pep 8**
  tres utile pour avoir un code propre, facile a lire, et a maintenir
  > regle syntaxique tel que :
  >  - nombre espace pour l'indentation
  >  - indentation des '}', ']', ')' sur plusieurs lignes
  >  - longueur maximal d'une ligne
  >  - regle de retour a la ligne : 
  >    - avant ou apres un operator
  >  - regle import
  >  - etc...

  https://www.flake8rules.com/
  https://peps.python.org/pep-0008/


- **pep 257**
  Documentation du code, norme simple d'ajout de Docstring, pour les classes, les fonctions
  > - Chaque classe d'un module, ainsi que ses methodes public doivent avoir un docstring
  > - les Docstrings doivent commencées par `"""` et finir par `"""`.
  > - ce sont des phrases qui explique ce que fait la fonctions.
  > - les Docstring multilique commence par `"""` suivi sur la meme ligne du résumé de ce que fait le fonction puis finissent par `"""` (seul sur une ligne).

  On peut également ameliorer cette norme en ajoutant les argument, les retours, les erreurs.
  > j'ai choisi le norme de Google

  exemple :
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