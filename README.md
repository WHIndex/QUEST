# QUEST

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

```bash
export MODEL_BASE_URL="https://example.com/v1"
export MODEL_API_KEY="your-key"
export MODEL_NAME="your-model"
```

Embedding service settings are optional. When omitted, lexical scoring is used.

```bash
export EMBED_BASE_URL="https://example.com/v1"
export EMBED_API_KEY="your-key"
export EMBED_MODEL="your-embedding-model"
```

## Data

Passages use JSON Lines format:

```json
{"id":"doc-1","title":"Document title","text":"Document text."}
```

Questions use JSON Lines format:

```json
{"id":"q-1","question":"Where was the author of The Hobbit born?"}
```

## Usage

Build an index:

```bash
quest build --passages data/passages.jsonl --index storage/example
```

Run one question:

```bash
quest ask --index storage/example --question "Where was the author of The Hobbit born?"
```

Run a question file:

```bash
quest run \
  --index storage/example \
  --questions data/questions.jsonl \
  --output outputs/results.jsonl
```

The batch command resumes from an existing output file by skipping completed identifiers.
