"""Core functionality for Educational Content Expander.

Milestone 1: Corpus gathering, preparation, and tokenization.
Milestone 2: N-gram model building with add-k smoothing.
Milestone 3: Text generation to expand short inputs.
Milestone 4: CLI integration helper.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# Default educational corpus covering diverse subjects (Biology, History, Physics, Math, Geography, Literature)
DEFAULT_EDUCATIONAL_CORPUS = """
Photosynthesis is the fundamental process used by plants and other organisms to convert light energy into chemical energy.
Through cellular respiration, cells break down glucose molecules to release energy in the form of adenosine triphosphate.
Deoxyribonucleic acid contains the hereditary genetic instructions used in the development and functioning of all known living organisms.
Ecosystems maintain biological balance through complex food webs involving primary producers, consumers, and decomposers.
Mitosis is a process of cell duplication where one cell divides into two genetically identical daughter cells.
Meiosis is a specialized type of cell division that reduces the chromosome number by half to produce gametes.
Natural selection is the differential survival and reproduction of individuals due to differences in phenotype.
Chloroplasts capture light energy from the sun to produce sugars and oxygen during plant photosynthesis.
The Industrial Revolution marked a major turning point in history with rapid technological growth and urbanization.
Ancient civilizations developed agriculture, writing systems, and social hierarchies along major river valleys.
The American Revolution led to the establishment of a constitutional democracy based on principles of liberty.
World War II was a global conflict that altered geopolitical alignments and led to the creation of the United Nations.
The Renaissance was a fervent period of cultural, artistic, political, and economic rebirth following the Middle Ages.
Isaac Newton formulated the universal laws of motion and the law of universal gravitation in classical mechanics.
Energy can neither be created nor destroyed, only transformed from one form into another according to thermodynamics.
Albert Einstein developed the theory of relativity which revolutionized our modern understanding of space, time, and gravity.
Electric current is the flow of electric charge carried by moving electrons through a conducting wire or circuit.
A linear equation represents a straight line on a Cartesian coordinate plane with a constant rate of change.
Quadratic equations can be solved using factoring, completing the square, or applying the quadratic formula.
Probability measures the likelihood of an event occurring and ranges from zero to one.
Plate tectonics explains how major landforms are created as a result of Earth subterranean subterranean movements.
Volcanic eruptions and earthquakes occur along the boundaries between dynamic tectonic plates.
The water cycle describes the continuous movement of water on, above, and below the surface of the Earth.
Evaporation, condensation, and precipitation are essential stages that sustain life across diverse biomes.
A metaphor creates a direct comparison between two unrelated concepts without using like or as.
Critical thinking enables students to analyze evidence objectively, formulate reasoned arguments, and evaluate claims.
Laboratory experiments allow students to test scientific hypotheses and observe empirical evidence directly.
Formative assessments provide ongoing feedback to improve teaching practices and enhance student learning outcomes.
Active learning strategies encourage students to participate directly in discussions, problem solving, and collaborative activities.
""".strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words/tokens, preserving contractions and punctuation where appropriate."""
    # Matches words with optional internal apostrophe, or standard sentence-boundary punctuation
    tokens = re.findall(r"\b\w+(?:'\w+)?\b|[.!?]", text.lower())
    return tokens


def prepare_corpus(text: str) -> List[List[str]]:
    """Split raw corpus into sentences and tokenize each sentence."""
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []
    for s in raw_sentences:
        tokens = tokenize(s)
        # Filter out standalone punctuation-only sentences
        clean_tokens = [t for t in tokens if t not in {".", "!", "?"}]
        if clean_tokens:
            sentences.append(clean_tokens)
    return sentences


def load_corpus(file_path: str | Path | None = None) -> List[List[str]]:
    """Load corpus from file or return the default educational corpus."""
    if file_path is not None:
        path = Path(file_path)
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return prepare_corpus(text)
    return prepare_corpus(DEFAULT_EDUCATIONAL_CORPUS)


class NGramModel:
    """N-gram language model with add-k smoothing."""

    def __init__(self, n: int = 3, k: float = 0.1):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.k = k
        self.vocab: set[str] = set()
        self.ngram_counts: Dict[Tuple[str, ...], int] = Counter()
        self.context_counts: Dict[Tuple[str, ...], int] = Counter()
        self.context_successors: Dict[Tuple[str, ...], set[str]] = defaultdict(set)
        self.total_tokens: int = 0

    def fit(self, sentences: Sequence[Sequence[str]]) -> None:
        """Train the n-gram model on tokenized sentences."""
        self.vocab = set()
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.context_successors.clear()
        self.total_tokens = 0

        pad = ["<s>"] * (self.n - 1)
        for sentence in sentences:
            if not sentence:
                continue
            for token in sentence:
                self.vocab.add(token)
                self.total_tokens += 1
            padded = pad + list(sentence) + ["</s>"]
            for i in range(len(padded) - self.n + 1):
                gram = tuple(padded[i : i + self.n])
                ctx = gram[:-1]
                target = gram[-1]
                self.ngram_counts[gram] += 1
                self.context_counts[ctx] += 1
                self.context_successors[ctx].add(target)

    def probability(self, word: str, context: Sequence[str]) -> float:
        """Calculate P(word | context) using add-k smoothing."""
        ctx = tuple(context[-(self.n - 1) :]) if self.n > 1 else ()
        gram = ctx + (word,)
        v = len(self.vocab) + 1  # include </s>
        gram_count = self.ngram_counts.get(gram, 0)
        ctx_count = self.context_counts.get(ctx, 0)

        # Add-k formula: (count(gram) + k) / (count(ctx) + k * |V|)
        if self.k > 0:
            return (gram_count + self.k) / (ctx_count + self.k * v)
        if ctx_count > 0:
            return gram_count / ctx_count
        return 1.0 / v if v > 0 else 0.0

    def next_word_candidates(
        self, context: Sequence[str], top_n: int = 10, exclude_boundary: bool = False
    ) -> List[Tuple[str, float]]:
        """Get candidate next words with their smoothed probabilities, sorted descending."""
        candidates = set()
        # Check contexts from full (n-1) order down to 0 order (backoff)
        for order in range(self.n - 1, -1, -1):
            ctx = tuple(context[-order:]) if order > 0 else ()
            if ctx in self.context_successors and self.context_successors[ctx]:
                candidates.update(self.context_successors[ctx])
                break

        if not candidates:
            candidates = set(self.vocab)
        else:
            candidates.add("</s>")

        if exclude_boundary:
            candidates.discard("</s>")
            candidates.discard("<s>")

        scored = [(word, self.probability(word, context)) for word in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_n]

    def predict_next(
        self,
        context: Sequence[str],
        strategy: str = "greedy",
        temperature: float = 1.0,
        exclude_boundary: bool = False,
    ) -> str:
        """Predict the single next word using greedy or sampling strategy."""
        candidates = self.next_word_candidates(
            context, top_n=20, exclude_boundary=exclude_boundary
        )
        if not candidates:
            return "."

        if strategy == "greedy":
            return candidates[0][0]

        # Sampling strategy with temperature
        words, probs = zip(*candidates)
        if temperature <= 0.01:
            return words[0]

        # Apply temperature
        log_probs = [math.log(max(p, 1e-12)) / temperature for p in probs]
        max_log = max(log_probs)
        exp_probs = [math.exp(lp - max_log) for lp in log_probs]
        sum_exp = sum(exp_probs)
        weights = [ep / sum_exp for ep in exp_probs]

        return random.choices(words, weights=weights, k=1)[0]


def expand_text(
    input_text: str,
    model: NGramModel,
    max_words: int = 15,
    strategy: str = "greedy",
    temperature: float = 0.7,
    seed: int | None = None,
) -> str:
    """Expand a short prompt or bullet point into a complete sentence using the n-gram model."""
    if seed is not None:
        random.seed(seed)
    clean_input = input_text.strip().lstrip("-*•0123456789. ")
    tokens = tokenize(clean_input)
    if not tokens:
        return ""

    context = list(tokens)
    pad = ["<s>"] * max(0, model.n - 1 - len(context))
    full_context = pad + context

    generated_tokens: List[str] = []
    for _ in range(max_words):
        # Prevent immediate sentence termination if we just started
        exclude_stop = len(generated_tokens) < 3
        next_word = model.predict_next(
            full_context,
            strategy=strategy,
            temperature=temperature,
            exclude_boundary=exclude_stop,
        )
        if next_word == "</s>":
            break
        generated_tokens.append(next_word)
        full_context.append(next_word)

    # Format result as a nicely capitalized and punctuated sentence
    all_words = tokens + generated_tokens
    expanded_sentence = " ".join(all_words)
    expanded_sentence = expanded_sentence[0].upper() + expanded_sentence[1:]
    if not expanded_sentence.endswith((".", "!", "?")):
        expanded_sentence += "."

    return expanded_sentence


# Milestone stubs required by the milestone roadmap:

def milestone_1(corpus_text: str | None = None) -> List[List[str]]:
    """Milestone 1: Gather a corpus of educational texts relevant to a subject.
    Hint: Corpus preparation and tokenization.
    """
    text = corpus_text if corpus_text is not None else DEFAULT_EDUCATIONAL_CORPUS
    sentences = prepare_corpus(text)
    return sentences


def milestone_2(
    sentences: Sequence[Sequence[str]] | None = None, n: int = 3, k: float = 0.1
) -> NGramModel:
    """Milestone 2: Build the n-gram model, including add-k smoothing.
    Hint: N-gram counting and probability calculation with add-k smoothing.
    """
    if sentences is None:
        sentences = milestone_1()
    model = NGramModel(n=n, k=k)
    model.fit(sentences)
    return model


def milestone_3(
    prompt: str,
    model: NGramModel | None = None,
    max_words: int = 12,
    strategy: str = "greedy",
) -> str:
    """Milestone 3: Implement text generation to expand short inputs.
    Hint: Greedy or sampling-based text generation.
    """
    if model is None:
        model = milestone_2()
    return expand_text(prompt, model, max_words=max_words, strategy=strategy)


def milestone_4(prompt: str, n: int = 3, k: float = 0.1, max_words: int = 12) -> str:
    """Milestone 4: Develop a simple CLI for text input and expansion output.
    Hint: Basic input/output handling for a command-line application.
    """
    model = milestone_2(n=n, k=k)
    return expand_text(prompt, model, max_words=max_words, strategy="greedy")
