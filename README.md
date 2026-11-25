# AI Project Intelligence & Risk Advisor

An end-to-end intelligent platform that ingests heterogeneous project artifacts (PDF, DOCX, CSV, TXT), constructs a unified vector knowledge base with RAG, runs a multi-agent analytical pipeline (Scope & Deliverables, Risk & Forecast, Blockers & Actions, Documentation Generation, Health Scoring), and exposes an interactive dashboard and grounded conversational assistant.

## Features & Modules

1. **Document Ingestion Engine**:
   - Parses `.pdf`, `.docx`, `.csv`, `.txt`, and `.md`.
   - Extracts structured metadata (source, title, timestamp, type).
2. **RAG Pipeline**:
   - Semantic & chunk-aware text splitting.
   - Vector indexing & retrieval (ChromaDB / Vector store with fallback embeddings).
   - Grounded context retrieval with source attribution.
3. **Multi-Agent Pipeline**:
   - **Scope & Deliverable Extraction Agent**: Extracts core objectives, deliverables, timelines, milestones, and owner mapping.
   - **Risk Detection & Delivery Forecasting Agent**: Assesses schedule slip risk, dependency bottlenecks, technical debt, and team capacity risks.
   - **Blocker & Action Item Agent**: Detects unresolved questions, blockers, and assigned action items with priority.
   - **Documentation Generation Agent**: Generates User Stories, Risk Registers, and Meeting Action Items.
   - **Project Health Scoring Engine**: Computes comprehensive health metric (0-100) across Scope, Schedule, Risk, and Velocity dimensions with qualitative grading.
4. **Interactive Dashboard & Conversational Assistant**:
   - Modern Streamlit UI with visual scorecards, risk matrices, generated documentation downloads, and chat assistant grounded entirely in uploaded project artifacts.

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```
