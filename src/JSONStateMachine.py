from enum import Enum, auto


class JSONState(Enum):
    START = auto()          # {
    PROMPT_KEY = auto()     # "prompt": "
    PROMPT_VAL = auto()     # [Valeur du prompt]
    NAME_KEY = auto()       # ", "name": "
    NAME_VAL = auto()       # [Nom de la fonction]
    PARAMS_KEY = auto()     # ", "parameters": {
    PARAM_NAME = auto()     # "nom_du_parametre"
    # PARAM_COLON = auto()    # :
    # PARAM_VAL = auto()      # [Valeur selon le type]
    # PARAM_COMMA = auto()    # ,
    END = auto()            # }


class JSONStateMachine:
    def __init__(self, model, functions_def, token_to_id, prompt=""):
        self.model = model
        self.state = JSONState.START
        self.buffer_tokens: list[int] = []
        self.current_text = ""

        self.functions_names = functions_def.list_functions_name()
        self.token_to_id = token_to_id

        # Targets encodés
        self.targets = {
            JSONState.START: model.encode("{")[0].tolist(),
            JSONState.PROMPT_KEY: model.encode('"prompt": "')[0].tolist(),
            JSONState.NAME_KEY: model.encode('", "name": "')[0].tolist(),
            JSONState.PARAMS_KEY: model.encode('", "parameters": {')[0].tolist(),
            JSONState.PROMPT_VAL: model.encode(prompt)[0].tolist(),
            # JSONState.PARAM_COLON: model.encode(": ")[0].tolist(),
            # JSONState.PARAM_COMMA: model.encode(", ")[0].tolist(),
            JSONState.END: model.encode('}')[0].tolist(),
        }

        self.progress = 0

    def get_target_tokens_for_current_state(self) -> list[int]:
        return self.targets.get(self.state, [])
    
    def is_in_fixed_sequence(self) -> bool:
        return self.state in self.targets

    def get_allowed_tokens(self) -> set[int]:
        # 1. Si on est dans une séquence fixe (JSON)
        if self.state in self.targets:
            target = self.targets[self.state]

            if self.progress < len(target):
                return {target[self.progress]}
        
        # 2. Cas dynamique
        if self.state == JSONState.NAME_VAL:
            return self._allowed_tokens_for_function_name()

        return set(range(self.token_to_id.__len__()))  # Tous les tokens sont autorisés par défaut
    
    def _allowed_tokens_for_function_name(self) -> set[int]:
        allowed_tokens = set()
        still_possible = [
            s for s in self.functions_names
            if s.startswith(self.current_text)
        ]
        for s in still_possible:
            allowed_tokens.update(
                self._get_allowed_tokens_for_string(
                    s,
                    self.current_text,
                    self.token_to_id
                )
            )
        return allowed_tokens

    def _get_allowed_tokens_for_string(
            self,
            target_string,
            current_generated_text,
            token_to_id
            ) -> set[int]:
        """
        Determine the next character that is expected
        and allow tokens that start with it.
        """
        # Si on a déjà fini la chaîne
        if current_generated_text == target_string:
            return {token_to_id.get(" ")}  # Ou un token de ponctuation comme ':'

        # Trouver ce qu'il reste à générer
        remaining = target_string[len(current_generated_text):]

        allowed = set()
        for token_str, t_id in token_to_id.items():
            # On autorise les tokens qui sont le début de ce qu'il reste à écrire
            # Attention : gérer les espaces de début de token (ex: 'Ġ' ou ' ')
            clean_token = token_str.replace('Ġ', ' ').replace(' ', ' ')
            if remaining.startswith(clean_token) and clean_token != "":
                allowed.add(t_id)
        return allowed

    def update(self, token_id: int):
        token_text = self.model.decode([token_id])

        self.buffer_tokens.append(token_id)
        self.current_text += token_text
        #print(f" -- FSM update: received token_id={token_id} ('{token_text}'), progress={self.progress}")  # Debug print
        # 1. Gestion des séquences fixes
        if self.state in self.targets:
            target = self.targets[self.state]

            if token_id == target[self.progress]:
                self.progress += 1

                if self.progress == len(target):
                    self._update_state()
                    self.current_text = ""
                    self.progress = 0
            else:
                raise ValueError("Invalid token in fixed sequence")

        # 2. Cas dynamique (nom de fonction)
        elif self.state == JSONState.NAME_VAL:
            if self.current_text in self.functions_names:
                self._update_state()
                self.current_text = ""

    def _update_state(self):
        if self.state == JSONState.START:
            self.state = JSONState.PROMPT_KEY
        elif self.state == JSONState.PROMPT_KEY:
            self.state = JSONState.PROMPT_VAL
        elif self.state == JSONState.PROMPT_VAL:
            self.state = JSONState.NAME_KEY
        elif self.state == JSONState.NAME_KEY:
            self.state = JSONState.NAME_VAL
        elif self.state == JSONState.NAME_VAL:
            self.state = JSONState.PARAMS_KEY
        elif self.state == JSONState.PARAMS_KEY:
            self.state = JSONState.PARAM_NAME
        # elif self.state == JSONState.PARAM_NAME:
        #     self.state = JSONState.PARAM_COLON
        # elif self.state == JSONState.PARAM_COLON:
        #     self.state = JSONState.PARAM_VAL
        # elif self.state == JSONState.PARAM_VAL:
        #     self.state = JSONState.PARAM_COMMA
        # elif self.state == JSONState.PARAM_COMMA:
        #     self.state = JSONState.PARAM_NAME  # ou END si pas de paramètre supplémentaire
        elif self.state == JSONState.PARAM_NAME:
            self.state = JSONState.END
        else:
            raise ValueError("Invalid state transition")  
