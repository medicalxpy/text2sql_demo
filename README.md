# Part1+Part2 Text-to-SQL Demo (Single-Version)

This directory contains a minimal CLI demo for:
- Part 1 (LLM + deterministic): SpecAgent (LLM intent parsing + deterministic gene normalization + deterministic topic retrieval)
- Part 2 (LLM): SQLGenAgent (K=1 candidate SQL)
- Part 3 (deterministic): Verifier/Selector (4-layer verification + dataset selection)

By default the CLI executes Part 1-3 and returns selected SQL + Top-K datasets.

## Prerequisites
- Python 3.10+
- `openai` Python client installed

## Configure the model endpoint
The code expects an OpenAI-compatible API.

You can place credentials in `text2sql_demo/.env.local` (auto-loaded by `llm_client.py`).
This file is gitignored.

Set environment variables:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8045/v1"
export OPENAI_API_KEY="..."
export LLM_MODEL="gemini-3-pro-high"
```

Notes:
- Do not hardcode API keys in source files.
- `OPENAI_BASE_URL` defaults to `http://127.0.0.1:8045/v1`.
- `LLM_MODEL` defaults to `gemini-3-pro-high`.

## Build topic store assets
This demo builds topic assets from an external gene-set library and anonymizes topic IDs as
`topic_1`, `topic_2`, ... (no source-specific topic names in generated outputs):
- `text2sql_demo/data/topic_gene_hallmark_2020.json`
- `text2sql_demo/data/topic_descriptions_hallmark_2020.json`

`build_topic_descriptions` always uses the LLM endpoint (no deterministic mode). Make sure
`OPENAI_API_KEY` and `LLM_MODEL` are configured before running it.

Run:

```bash
python -m text2sql_demo.scripts.build_topic_store
python -m text2sql_demo.scripts.build_topic_descriptions
```

## Build full simulated 5-table store
This builds a full synthetic retrieval store with the canonical 5 tables:
- `topic_gene`
- `topic_description`
- `dataset`
- `cell`
- `cell_topic`

Default config:
- `text2sql_demo/config/sim_store_v1.json`

Run:

```bash
python -m text2sql_demo.scripts.build_simulated_store
```

Optional: regenerate topic assets before full-store build (requires LLM credentials):

```bash
python -m text2sql_demo.scripts.build_simulated_store --refresh-topic-assets
```

Generated artifacts:
- `text2sql_demo/data/sim_store_v1.sqlite`
- `text2sql_demo/data/sim_store_v1.sql`
- `text2sql_demo/data/sim_store_v1_manifest.json`

Validate the generated SQLite store:

```bash
python -m text2sql_demo.scripts.validate_simulated_store
```

Validation report:
- `text2sql_demo/data/sim_store_v1_validation.json`

## Run the CLI

```bash
python -m text2sql_demo
```

Type a question in English and press enter. Type `exit` to quit.

Part1+Part2 only (no SQLite execution):

```bash
python -m text2sql_demo --no-execute
```

Select method:

- `workflow` (default): SpecAgent (LLM parse + gene normalization + topic grounding) + SQLGenAgent
- `baseline_a`: single-shot LLM SQL generation (question -> SQL candidates)
- `baseline_b`: LLM topic selection + SQLGen (no deterministic topic grounding)

Examples:

```bash
python -m text2sql_demo --mode workflow
python -m text2sql_demo --mode baseline_a
python -m text2sql_demo --mode baseline_b
```

Part1+Part2 only for each method:

```bash
python -m text2sql_demo --mode workflow --no-execute
python -m text2sql_demo --mode baseline_a --no-execute
python -m text2sql_demo --mode baseline_b --no-execute
```

Run Part3 against a specific SQLite file:

```bash
python -m text2sql_demo --mode workflow --db-path text2sql_demo/data/sim_store_v1.sqlite --query-timeout-seconds 2.0
python -m text2sql_demo --mode baseline_a --db-path text2sql_demo/data/sim_store_v1.sqlite --query-timeout-seconds 2.0
python -m text2sql_demo --mode baseline_b --db-path text2sql_demo/data/sim_store_v1.sqlite --query-timeout-seconds 2.0
```
