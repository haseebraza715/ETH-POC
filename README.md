# Claims PoC (LangGraph + Python)

This proof-of-concept demonstrates a local claims intake agent that:

- chats with the operator to collect missing claim details
- ingests a police report (txt or pdf) and extracts structured data via LLM (or fallback heuristics)
- cross-checks user input against document fields, highlighting gaps or mismatches
- loops until the claim is complete and consistent, then finalizes with a summary + reasoning trace

Everything runs locally using LangGraph for orchestration.

## Project layout

```
claims_poc/
  main.py              # CLI entrypoint
  graph.py             # LangGraph wiring + nodes
  state.py             # ClaimState dataclass
  llm_client.py        # LLM helper with rule-based fallback
  tools/
    schema.py          # required fields per claim type
    doc_parser.py      # txt/PDF ingestion
    extractor.py       # LLM-powered extraction
    consistency.py     # completeness + mismatch checks
  sample_data/
    police_report_example.txt
    police_report_example.pdf
```

## Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key libraries:

- `langgraph`, `langchain-core`, `langchain-openai`
- `pypdf` for PDF parsing

> **LLM access:**  
> - Set `OPENROUTER_API_KEY` to call OpenRouter (default model `mistralai/mistral-small-3.2-24b-instruct:free`). Optional helpers: `OPENROUTER_MODEL`, `OPENROUTER_SITE_URL`, `OPENROUTER_SITE_NAME`.  
> - Alternatively, set `OPENAI_API_KEY` for direct OpenAI access.  
> - Without either, the system uses the deterministic rule-based fallback so you can still run the demo end-to-end.

## Prompt pack

Every LLM call uses a shared system prompt (“insurance claims assistant, precise, no guessing”) plus focused user prompts:

- **Document extraction:** strict JSON instructions (date/time/location/plate/injuries/description) with “no guessing” reminder.
- **Question generation:** given the current `ClaimState` + required schema, the model proposes up to three concise questions to collect missing data.
- **Inconsistency clarification:** politely cites both conflicting values (user vs. document) and asks for confirmation.
- **Final summary:** 120–200 word handler brief covering what/when/where/who/injuries/damage + missing info notes.
- **Reasoning trace:** converts internal event logs into 5–10 bullet points so ETH/Zürich reviewers see what the agent did.

If no remote LLM is configured, deterministic fallbacks craft plain questions/summaries so the workflow still completes.

## Running the demo

From the repo root:

```bash
python -m claims_poc.main
```

The CLI will:

1. ask for required claim fields (date, time, etc.)
2. request a police report path – press Enter to use the sample text file
3. parse & extract document facts
4. loop to resolve missing fields or inconsistencies
5. print the final structured claim, handler summary, polished reasoning trace, plus the raw internal event log for auditing

You can skip the document prompt by providing `--doc`:

```bash
python -m claims_poc.main --doc claims_poc/sample_data/police_report_example.pdf
```

## Sample data

- `sample_data/police_report_example.txt` – human-readable report
- `sample_data/police_report_example.pdf` – tiny PDF with the same content (parsed via `pypdf`; if parsing fails on your platform the parser gracefully falls back to raw text decoding)

Feel free to replace these with your own documents; the doc parser simply needs a filesystem path.

# ETH-POC
