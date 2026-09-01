import asyncio

from textual.widgets import DataTable, Static

from src.generator import GenerationStep, TokenCandidate
from src.visualizer import GenerationVisualizer


def test_visualizer_renders_generation_step(monkeypatch):
    """Exercise the dashboard headlessly without loading the real model."""
    monkeypatch.setattr(
        GenerationVisualizer,
        "_generate_all",
        lambda self: None,
    )
    app = GenerationVisualizer("functions.json", "prompts.json", None)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app._populate_prompts(["Toggle the feature"])
            app._start_prompt(0, "Toggle the feature")
            app._apply_step(GenerationStep(
                index=5,
                state="NAME_VAL",
                kind="model",
                selected_id=42,
                selected_text="toggle",
                generated_text='{"name": "toggle',
                response_tokens=7,
                allowed_count=2,
                candidates=(
                    TokenCandidate(42, "toggle", 8.5),
                    TokenCandidate(43, "greet", 7.1),
                ),
            ))
            await pilot.pause()

            assert "NAME_VAL" in str(
                app.query_one("#state", Static).content
            )
            assert "toggle" in str(
                app.query_one("#json-output", Static).content
            )
            assert app.query_one("#candidates", DataTable).row_count == 2

    asyncio.run(exercise())


def test_token_preview_never_exceeds_candidate_column():
    preview = GenerationVisualizer._token_preview("x" * 100)

    assert len(preview) == 14
    assert preview.endswith("…")
    assert GenerationVisualizer._token_preview("ok") == "'ok'"
