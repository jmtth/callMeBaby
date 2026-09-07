from pathlib import Path
import sys
import json


def save_results_to_json(
          results: list[dict[str, object]],
          output_path: str | None
          ) -> None:
    """Save results to a JSON file, creating directories if needed.

    Args:
        results: The results dictionary to save.
        output_path: The path to the output JSON file.
        If None, no file is saved.
    """
    if output_path is not None:
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(results,
                                              indent=2,
                                              ensure_ascii=False
                                              ), encoding="utf-8")
        except Exception as exc:
            print(
                f"Error writing output file {output_path}: {exc}",
                file=sys.stderr
                )
            new_file_name = input("Enter a new output file name: ")
            output_path = f"data/output/{new_file_name}.json"
            try:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(json.dumps(results,
                                                  indent=2,
                                                  ensure_ascii=False
                                                  ), encoding="utf-8")
            except Exception as exc:
                print(
                    f"Error writing output file {output_path}: {exc}",
                    file=sys.stderr
                    )
                print("Failed to save results to JSON file.", file=sys.stderr)
                raise
