# Educational Content Expander

A CLI tool designed for educators to expand concise lesson plan bullet points and short phrases into complete, grammatically sound, and contextually relevant instructional sentences using an $N$-gram language model with add-$k$ smoothing.

---

## Features

- **Bullet Point & Short Phrase Expansion**: Automatically strips list bullets (`-`, `*`, `•`, numbered lists) and continues the thought using statistical language modeling.
- **$N$-Gram Language Modeling with Add-$k$ Smoothing**:
  - Configurable $n$-gram order (e.g. bigrams, trigrams).
  - Add-$k$ Laplace smoothing ($P(w | context) = \frac{\text{count} + k}{\text{ctx\_count} + k \cdot |V|}$) to avoid zero-probability bottlenecks.
  - Multi-order backoff support for context-aware transitions.
- **Flexible Decoding Strategies**:
  - **Greedy Generation**: Picks the most probable sequence deterministically.
  - **Temperature-Scaled Sampling**: Introduces controlled stochasticity for varied educational phrasing.
- **Interactive Suggestions (`suggest`)**: View ranked next-word candidates and their calculated probability distribution.
- **Custom Corpus Support**: Train on custom educational textbooks, curricula, or notes, or use the built-in cross-disciplinary corpus (biology, history, physics, mathematics, literature, pedagogy).
- **File Export**: Directly export generated sentences to lesson plan drafts or note files using `--output`.

---

## Project Structure

```
├── app/
│   ├── cli.py          # Command-line interface and argument parsing
│   ├── core.py         # N-gram model, add-k smoothing, tokenization, text expansion
│   ├── llm.py          # LLM integration client (Ollama/Groq/Gemini compatibility)
│   └── storage.py      # Lightweight JSON store
├── tests/
│   ├── test_core.py    # Unit tests for milestones 1 through 4
│   └── test_smoke.py   # Baseline system checks
├── pytest.ini          # Pytest configuration
├── requirements.txt    # Project dependencies
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.10+ (recommended: Python 3.11 or 3.12)
- Git

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd educational-content-expander

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## CLI Usage

Run the tool using `python -m app.cli [COMMAND] [OPTIONS]`.

### 1. Expand a Bullet Point into a Sentence

Expand concise notes into full lesson plan statements:

```bash
python -m app.cli expand "Photosynthesis is the fundamental"
```
*Output:*
```text
Photosynthesis is the fundamental process used by plants and other organisms to convert light.
```

#### Common Options for `expand`:

| Option | Type | Default | Description |
|---|---|---|---|
| `text` | `string` | *(required)* | Input bullet point or seed phrase |
| `--n` | `int` | `3` | $N$-gram order |
| `--k` | `float` | `0.1` | Add-$k$ smoothing parameter |
| `--max-words` | `int` | `12` | Maximum generated tokens |
| `--strategy` | `greedy` \| `sampling` | `greedy` | Generation algorithm |
| `--temperature` | `float` | `0.7` | Softmax temperature for sampling |
| `--corpus` | `path` | `None` | Path to custom `.txt` corpus |
| `--output` | `path` | `None` | File path to write output sentence |

#### Examples:

**Sampling with temperature:**
```bash
python -m app.cli expand "• Active learning strategies" --strategy sampling --temperature 0.8 --max-words 10
```

**Export directly to a lesson plan file:**
```bash
python -m app.cli expand "- Formative assessments provide" --output lesson_plan.txt
```

**Using a custom curriculum text file:**
```bash
python -m app.cli expand "Plate tectonics" --corpus path/to/geology_unit.txt --n 3
```

---

### 2. Next-Word Suggestions

Inspect probability distributions and top predicted words for any phrase:

```bash
python -m app.cli suggest "Photosynthesis is" --top 5
```
*Output:*
```text
Top 1 suggestions for 'Photosynthesis is':
  1. the (P = 0.0345)
```

---

### 3. Verify Setup

Check CLI plumbing:
```bash
python -m app.cli hello
```

---

## Running Tests

Run the full test suite with `pytest`:

```bash
pytest -v
```

All core milestone validations and smoke tests will run and report status.
