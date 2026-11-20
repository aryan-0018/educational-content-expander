"""Integration and smoke tests for Educational Content Expander."""

from app.cli import main


def test_the_command_line_runs(capsys):
    main(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_the_command_line_expand(capsys):
    main(["expand", "Photosynthesis is", "--max-words", "5"])
    output = capsys.readouterr().out
    assert "photosynthesis is" in output.lower()


def test_the_command_line_suggest(capsys):
    main(["suggest", "Photosynthesis is", "--top", "3"])
    output = capsys.readouterr().out
    assert "top" in output.lower()
    assert "photosynthesis is" in output.lower()
