"""Tests for Educational Content Expander milestones."""

import pytest
from app.core import (
    DEFAULT_EDUCATIONAL_CORPUS,
    NGramModel,
    expand_text,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    prepare_corpus,
    tokenize,
)


# Milestone 1: Gather a corpus of educational texts relevant to a subject.
def test_milestone_1():
    sentences = milestone_1()
    assert len(sentences) >= 10
    # Every sentence should be tokenized into words
    assert all(isinstance(s, list) for s in sentences)
    assert all(len(s) > 0 for s in sentences)
    # Check tokenization of sample sentence
    tokens = tokenize("Photosynthesis converts light energy into chemical energy.")
    assert "photosynthesis" in tokens
    assert "chemical" in tokens


# Milestone 2: Build the n-gram model, including add-k smoothing.
def test_milestone_2():
    corpus = [
        ["photosynthesis", "produces", "glucose", "and", "oxygen"],
        ["chloroplasts", "enable", "photosynthesis", "in", "plant", "cells"],
        ["cells", "break", "down", "glucose", "for", "energy"],
    ]
    model = NGramModel(n=3, k=0.1)
    model.fit(corpus)

    # Check vocabulary and counts
    assert "photosynthesis" in model.vocab
    assert "glucose" in model.vocab

    # Check smoothed probability
    prob_known = model.probability("glucose", ["photosynthesis", "produces"])
    assert prob_known > 0.0

    # Add-k smoothing ensures unseen word in context has non-zero probability
    prob_unseen = model.probability("unseen_word", ["photosynthesis", "produces"])
    assert prob_unseen > 0.0
    assert prob_known > prob_unseen

    # Milestone 2 factory returns a trained model
    m2_model = milestone_2()
    assert isinstance(m2_model, NGramModel)
    assert len(m2_model.vocab) > 50


# Milestone 3: Implement text generation to expand short inputs.
def test_milestone_3():
    model = milestone_2(n=2, k=0.05)
    expansion = milestone_3("Photosynthesis is", model=model, max_words=6, strategy="greedy")
    assert isinstance(expansion, str)
    assert expansion.lower().startswith("photosynthesis is")
    assert len(expansion.split()) > 2

    # Sampling strategy test
    sampled = milestone_3("Photosynthesis", model=model, max_words=5, strategy="sampling")
    assert isinstance(sampled, str)
    assert len(sampled.split()) > 1


# Milestone 4: Develop a simple CLI for text input and expansion output.
def test_milestone_4(tmp_path):
    # Test basic expansion
    result = milestone_4("Natural selection")
    assert isinstance(result, str)
    assert "natural selection" in result.lower()
    assert result.endswith((".", "!", "?"))

    # Test milestone_4 with full option suite and file output
    out_file = str(tmp_path / "expanded_lesson.txt")
    result_saved = milestone_4(
        "• Photosynthesis is",
        n=2,
        k=0.05,
        max_words=8,
        strategy="sampling",
        seed=100,
        output=out_file,
    )
    assert "photosynthesis is" in result_saved.lower()
    from pathlib import Path
    assert Path(out_file).read_text(encoding="utf-8").strip() == result_saved
