"""The examples run, and the README names the Luma this build is compiled against."""

import runpy
from pathlib import Path

import pytest

import lumapy as lp

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(path, capsys):
    runpy.run_path(str(path), run_name="__main__")
    assert capsys.readouterr().out


def test_readme_names_the_mirrored_luma():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"Luma {lp.__luma_version__}" in readme
    if lp.__luma_source__ == "git":
        assert lp.__luma_commit__[:7] in readme, "update the 'Mirrors Luma' line of the README"
