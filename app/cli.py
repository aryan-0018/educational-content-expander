"""Command-line entry point: `python -m app.cli --help`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.core import expand_text, load_corpus, milestone_2, milestone_4


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="educational-content-expander",
        description="Educators use this CLI tool to expand bullet points or short phrases into coherent sentences for lesson plans by generating next-word suggestions based on educational texts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # hello command
    sub.add_parser("hello", help="check that everything is wired up")

    # expand command
    expand_p = sub.add_parser(
        "expand",
        help="Expand a short phrase or bullet point into a coherent sentence",
    )
    expand_p.add_argument("text", help="The bullet point or phrase to expand")
    expand_p.add_argument(
        "--n", type=int, default=3, help="N-gram order (default: 3)"
    )
    expand_p.add_argument(
        "--k", type=float, default=0.1, help="Add-k smoothing parameter (default: 0.1)"
    )
    expand_p.add_argument(
        "--max-words",
        type=int,
        default=12,
        help="Maximum additional words to generate (default: 12)",
    )
    expand_p.add_argument(
        "--strategy",
        choices=["greedy", "sampling"],
        default="greedy",
        help="Generation strategy: greedy or sampling (default: greedy)",
    )
    expand_p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature when using sampling strategy (default: 0.7)",
    )
    expand_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: None)",
    )
    expand_p.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Optional path to custom educational corpus text file",
    )
    expand_p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to save the expanded content",
    )

    # suggest command (stretch idea: show candidate next words)
    suggest_p = sub.add_parser(
        "suggest",
        help="Suggest the top next-word predictions for a given context",
    )
    suggest_p.add_argument("text", help="Context text to suggest next words for")
    suggest_p.add_argument(
        "--top", type=int, default=5, help="Number of suggestions (default: 5)"
    )
    suggest_p.add_argument(
        "--n", type=int, default=3, help="N-gram order (default: 3)"
    )
    suggest_p.add_argument(
        "--k", type=float, default=0.1, help="Add-k smoothing parameter (default: 0.1)"
    )

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Educational Content Expander is ready. Open README.md for your milestones.")
    elif args.command == "expand":
        expanded = milestone_4(
            prompt=args.text,
            n=args.n,
            k=args.k,
            max_words=args.max_words,
            strategy=args.strategy,
            temperature=args.temperature,
            seed=args.seed,
            output=args.output,
            corpus_path=args.corpus,
        )
        print(expanded)
        if args.output:
            print(f"(Saved expansion to {args.output})")
    elif args.command == "suggest":
        sentences = load_corpus()
        model = milestone_2(sentences, n=args.n, k=args.k)
        from app.core import tokenize

        tokens = tokenize(args.text)
        candidates = model.next_word_candidates(tokens, top_n=args.top, exclude_boundary=True)
        print(f"Top {len(candidates)} suggestions for '{args.text}':")
        for rank, (word, prob) in enumerate(candidates, 1):
            print(f"  {rank}. {word} (P = {prob:.4f})")


if __name__ == "__main__":
    main()
