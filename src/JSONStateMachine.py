from enum import Enum, auto
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Optional

class JSONState(Enum):
    START = auto()           # Avant le '{'
    OBJECT_OPEN = auto()     # Après le '{'
    KEY_NAME = auto()        # En train de générer une clé ("prompt", "name", etc.)
    COLON = auto()           # Après une clé, attend ':'
    VALUE_STRING = auto()    # En train de générer une chaîne
    VALUE_NUMBER = auto()    # En train de générer un nombre [cite: 319]
    VALUE_OBJECT = auto()    # Pour les "parameters" [cite: 303]
    COMMA = auto()           # Entre deux paires clé-valeur
    END = auto()             # Après le '}' final

class JSONStateMachine:
    def __init__(self, schema: dict, vocabulary: dict):
        self.schema = schema
        self.vocab = vocabulary # Charge le vocabulary.json [cite: 252, 293]
        self.current_buffer = ""
        # États possibles : START, IN_KEY, AFTER_KEY, IN_VALUE, etc.
        self.state = "START" 

    def get_allowed_tokens(self) -> set[int]:
        """
        Analyse self.current_buffer pour déterminer les tokens valides.
        C'est ici que réside la complexité du projet.
        """
        allowed_ids = set()
        # Logique de filtrage basée sur la structure JSON et le schéma [cite: 285]
        # Exemple : si l'état est START, seul le token '{' est autorisé.
        return allowed_ids

    def update_state(self, last_token_text: str):
        self.current_buffer += last_token_text
        # Mise à jour de l'état interne selon le texte ajouté