"""The worked example runs and reports the published PFOS muscle TFL."""

import importlib.util
from pathlib import Path

EXAMPLE = Path(__file__).parents[1] / "examples" / "worked_example.py"


def test_worked_example(capsys):
    spec = importlib.util.spec_from_file_location("worked_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main("PFOS", feed=1.0, stop_day=25, n_samples=20)
    assert "muscle   TFL = 0.528" in capsys.readouterr().out
    module.main("PFDA", feed=1.0, stop_day=25, n_samples=20)  # no EU maximum level
    assert "no tolerable feed level" in capsys.readouterr().out
