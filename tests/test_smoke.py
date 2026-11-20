"""Integration and smoke tests for Educational Content Expander."""

from app.cli import main
from app.storage import Store


def test_the_store_keeps_what_you_put_in_it(tmp_path):
    store = Store(str(tmp_path / "app.json"))
    store.put("hello", {"n": 1})
    assert store.get("hello") == {"n": 1}
    assert store.get("missing", "fallback") == "fallback"
    assert store.all() == {"hello": {"n": 1}}


def test_the_command_line_runs(capsys):
    main(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_the_command_line_expand(capsys):
    main(["expand", "Photosynthesis is", "--max-words", "5"])
    output = capsys.readouterr().out
    assert "photosynthesis is" in output.lower()
