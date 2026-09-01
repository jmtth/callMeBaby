"""Textual interface for observing constrained token generation."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from src.call_me_maybe import (
    FunctionsDefinition,
    generate_response,
    load_model,
    load_prompts,
    validate_generated_response,
)
from src.generator import GenerationStep
from src.token_vocabulary import TokenVocabulary


class GenerationVisualizer(App[None]):
    """Interactive dashboard for constrained-decoding decisions."""

    TITLE = "Call Me Maybe - Token Generation Lab"
    SUB_TITLE = "Constrained decoding, one decision at a time"

    CSS = """
    Screen {
        background: #07111f;
        color: #d8e8ff;
    }

    Header {
        background: #10243d;
        color: #f5fbff;
    }

    #workspace {
        height: 1fr;
    }

    #sidebar {
        width: 31;
        min-width: 24;
        border-right: solid #1f6f8b;
        background: #0b1829;
    }

    .section-title {
        height: 3;
        padding: 1 2 0 2;
        color: #5eead4;
        text-style: bold;
    }

    #prompt-list {
        height: 1fr;
        padding: 0 1;
        background: #0b1829;
    }

    ListItem {
        padding: 1;
    }

    ListItem.--highlight {
        background: #12395a;
    }

    #sidebar-help {
        height: 5;
        padding: 1 2;
        color: #7896b8;
        border-top: solid #17324f;
    }

    #content {
        width: 1fr;
        padding: 1 2;
    }

    #prompt-card {
        height: auto;
        min-height: 5;
        padding: 1 2;
        margin-bottom: 1;
        border: round #245f86;
        background: #0d1d30;
    }

    #metrics {
        height: 3;
        margin-bottom: 1;
    }

    .metric {
        width: 1fr;
        padding: 0 1;
        border-left: thick #2dd4bf;
        background: #10243d;
        content-align: left middle;
    }

    #middle {
        height: 1fr;
        min-height: 18;
    }

    #json-column {
        width: 3fr;
        margin-right: 1;
    }

    #choice-column {
        width: 2fr;
    }

    .panel-title {
        height: 2;
        color: #93c5fd;
        text-style: bold;
    }

    #json-output {
        height: 1fr;
        padding: 1 2;
        border: round #245f86;
        background: #091524;
    }

    #candidates {
        height: 1fr;
        border: round #245f86;
        background: #091524;
    }

    #timeline-title {
        margin-top: 1;
    }

    #timeline {
        height: 9;
        border: round #245f86;
        background: #091524;
        padding: 0 1;
    }

    Footer {
        background: #10243d;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quitter"),
        ("c", "clear_timeline", "Effacer le journal"),
    ]

    def __init__(
        self,
        functions_path: str,
        input_path: str | None,
        output_path: str | None,
        max_response_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.functions_path = functions_path
        self.input_path = input_path
        self.output_path = output_path
        self.max_response_tokens = max_response_tokens
        self._prompt_labels: list[Label] = []
        self._started_at = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="sidebar"):
                yield Static("PROMPTS", classes="section-title")
                yield ListView(id="prompt-list")
                yield Static(
                    "Le bleu indique un choix du modèle.\n"
                    "Le vert indique une contrainte déterministe.",
                    id="sidebar-help",
                )
            with Vertical(id="content"):
                yield Static("Chargement des données...", id="prompt-card")
                with Horizontal(id="metrics"):
                    yield Static("ÉTAT\n—", id="state", classes="metric")
                    yield Static("ÉTAPE\n0", id="step", classes="metric")
                    yield Static("MODE\n—", id="mode", classes="metric")
                    yield Static(
                        "TOKENS\n0",
                        id="token-count",
                        classes="metric",
                    )
                with Horizontal(id="middle"):
                    with Vertical(id="json-column"):
                        yield Static(
                            "JSON EN CONSTRUCTION",
                            classes="panel-title",
                        )
                        yield Static("{}", id="json-output")
                    with Vertical(id="choice-column"):
                        yield Static(
                            "TOKENS AUTORISÉS - TOP 20",
                            classes="panel-title",
                        )
                        yield DataTable(id="candidates", zebra_stripes=True)
                yield Static(
                    "JOURNAL DE GÉNÉRATION",
                    id="timeline-title",
                    classes="panel-title",
                )
                yield RichLog(id="timeline", markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#candidates", DataTable)
        table.add_column("", width=1)
        table.add_column("Token", width=19)
        table.add_column("ID", width=8)
        table.add_column("Logit", width=10)
        table.cursor_type = "row"
        self._generate_all()

    @staticmethod
    def _token_preview(token_text: str, width: int = 14) -> str:
        """Return a stable-width representation for the candidate table."""
        rendered = repr(token_text)
        if len(rendered) <= width:
            return rendered
        return rendered[:width - 1] + "…"

    def action_clear_timeline(self) -> None:
        self.query_one("#timeline", RichLog).clear()

    def _set_status(self, message: str) -> None:
        self.query_one("#prompt-card", Static).update(message)

    def _populate_prompts(self, prompts: list[str]) -> None:
        prompt_list = self.query_one("#prompt-list", ListView)
        for index, prompt in enumerate(prompts, start=1):
            label = Label(f"○ {index:02d}  {prompt}")
            self._prompt_labels.append(label)
            prompt_list.append(ListItem(label))

    def _start_prompt(self, index: int, prompt: str) -> None:
        self._started_at = perf_counter()
        self.query_one("#prompt-list", ListView).index = index
        self._prompt_labels[index].update(f"▶ {index + 1:02d}  {prompt}")
        self.query_one("#prompt-card", Static).update(
            Text.assemble(
                (f"PROMPT {index + 1}\n", "bold #5eead4"),
                prompt,
            )
        )
        self.query_one("#json-output", Static).update("{}")
        self.query_one("#candidates", DataTable).clear()
        self.query_one("#token-count", Static).update("TOKENS\n0")
        self.query_one("#timeline", RichLog).write(
            f"[bold #5eead4]Prompt {index + 1}[/]  {prompt}"
        )

    def _finish_prompt(self, index: int, prompt: str, response: str) -> None:
        elapsed = perf_counter() - self._started_at
        self._prompt_labels[index].update(f"✓ {index + 1:02d}  {prompt}")
        try:
            formatted = json.dumps(
                json.loads(response),
                indent=2,
                ensure_ascii=False,
            )
        except json.JSONDecodeError:
            formatted = response
        self.query_one("#json-output", Static).update(formatted)
        self.query_one("#timeline", RichLog).write(
            f"[bold #34d399]Réponse validée[/] en {elapsed:.2f} s"
        )

    def _fail(self, message: str) -> None:
        self.query_one("#prompt-card", Static).update(
            Text(message, style="bold #fb7185")
        )
        self.query_one("#timeline", RichLog).write(
            f"[bold #fb7185]Erreur[/] {message}"
        )

    def _apply_step(self, event: GenerationStep) -> None:
        kind_labels = {
            "model": "CHOIX LLM",
            "fixed": "STRUCTURE",
            "deterministic": "CONTRAINTE",
        }
        colors = {
            "model": "#60a5fa",
            "fixed": "#a78bfa",
            "deterministic": "#34d399",
        }
        color = colors[event.kind]
        self.query_one("#state", Static).update(f"ÉTAT\n{event.state}")
        self.query_one("#step", Static).update(f"ÉTAPE\n{event.index}")
        self.query_one("#mode", Static).update(
            f"MODE\n{kind_labels[event.kind]}"
        )
        self.query_one("#token-count", Static).update(
            f"TOKENS\n{event.response_tokens}"
        )
        self.query_one("#json-output", Static).update(event.generated_text)

        table = self.query_one("#candidates", DataTable)
        table.clear()
        if event.candidates:
            for rank, candidate in enumerate(event.candidates):
                marker = "●" if candidate.token_id == event.selected_id else ""
                style = f"bold {color}" if rank == 0 else "#9ab2cc"
                table.add_row(
                    Text(marker, style=style),
                    Text(self._token_preview(candidate.text), style=style),
                    str(candidate.token_id),
                    f"{candidate.logit:.4f}",
                )
        else:
            table.add_row(
                Text("●", style=f"bold {color}"),
                Text(
                    self._token_preview(event.selected_text),
                    style=f"bold {color}",
                ),
                "—" if event.selected_id is None else str(event.selected_id),
                "déterministe",
            )

        token = event.selected_text.replace("\n", "\\n")
        self.query_one("#timeline", RichLog).write(
            f"[{color}]#{event.index:03d} {event.state:<12} "
            f"{kind_labels[event.kind]:<11} {token!r}[/]"
        )

    @work(thread=True, exclusive=True)
    def _generate_all(self) -> None:
        try:
            functions = FunctionsDefinition.from_json(self.functions_path)
            prompts = load_prompts(self.input_path)
            self.call_from_thread(self._populate_prompts, prompts)
            self.call_from_thread(self._set_status, "Chargement du modèle...")
            llm = load_model()
            vocabulary = TokenVocabulary(llm[0], llm[1])
            results: list[dict[str, object]] = []

            for index, prompt in enumerate(prompts):
                self.call_from_thread(self._start_prompt, index, prompt)

                def observe(event: GenerationStep) -> None:
                    self.call_from_thread(self._apply_step, event)

                response = generate_response(
                    functions,
                    prompt,
                    llm=llm,
                    max_res_tokens=self.max_response_tokens,
                    vocabulary=vocabulary,
                    observer=observe,
                )
                result = validate_generated_response(
                    functions,
                    prompt,
                    response,
                )
                results.append(result)
                self.call_from_thread(
                    self._finish_prompt,
                    index,
                    prompt,
                    response,
                )

            if self.output_path is not None:
                output = Path(self.output_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(results, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            self.call_from_thread(
                self._set_status,
                f"{len(results)} réponse(s) générée(s) et validée(s).",
            )
        except Exception as exc:
            self.call_from_thread(self._fail, str(exc))


def run_visualizer(
    functions_path: str,
    input_path: str | None,
    output_path: str | None,
) -> None:
    """Launch the interactive generation visualizer."""
    GenerationVisualizer(functions_path, input_path, output_path).run()
