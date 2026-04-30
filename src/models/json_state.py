from enum import Enum, auto


class JSONState(Enum):
    START = auto()          # {
    PROMPT_KEY = auto()     # "prompt": "
    PROMPT_VAL = auto()     # [Valeur du prompt]
    NAME_KEY = auto()       # ", "name": "
    NAME_VAL = auto()       # [Nom de la fonction]
    PARAMS_KEY = auto()     # ", "parameters": {
    PARAM_NAME = auto()     # "nom_du_parametre"
    PARAM_COLON = auto()    # :
    PARAM_VAL = auto()      # [Valeur selon le type]
    PARAM_COMMA = auto()    # ,
    END = auto()            # }}
    STOP = auto()           # Token de fin de génération
