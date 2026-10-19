import ast
from pathlib import Path


def _preset_names():
    source = Path(__file__).parents[1] / "mallinella_generator.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PRESETS"
            for target in node.targets
        ):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError("PRESETS definition not found")


def test_generic_ctenidae_presets_are_available():
    names = _preset_names()
    assert {
        "ctenus_generic",
        "macroctenus_generic",
        "amicactenus_generic",
    } <= names
