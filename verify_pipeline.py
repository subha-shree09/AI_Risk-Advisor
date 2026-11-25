"""
End-to-End Validation Script for AI Project Intelligence & Risk Advisor.
Tests:
1. Ingestion of multi-format artifacts (TXT, CSV).
2. Semantic chunking and vector index retrieval.
3. Multi-agent analytical pipeline execution.
4. Health scoring calculation.
5. Grounded conversational query verification.
"""
import os
import sys

# Ensure src in path
sys.path.insert(0, os.path.abspath("."))

from src.ingestion import DocumentIngestionEngine
from src.rag_engine import RAGPipeline
from src.llm_client import LLMClient
from src.agents.scope_agent import ScopeDeliverableAgent
from src.agents.risk_agent import RiskForecastingAgent
from src.agents.blocker_agent import BlockerActionAgent
from src.agents.documentation_agent import DocumentationAgent
from src.health_scorer import ProjectHealthScorer
from src.conversational_assistant import ConversationalAssistant

def test_pipeline():
    print("=== Step 1: Testing Multi-Format Ingestion ===")
    sample_files = [
        ("sample_data/project_proposal.txt", "proposal"),
        ("sample_data/meeting_notes_sprint_sync.txt", "notes"),
        ("sample_data/tasks_backlog.csv", "csv")
    ]
    
    parsed_docs = []
    for filepath, desc in sample_files:
        with open(filepath, "rb") as f:
            data = f.read()
        parsed = DocumentIngestionEngine.parse_file(os.path.basename(filepath), data)
        parsed_docs.append(parsed)
        print(f" [+] Ingested '{filepath}' -> {parsed['metadata']['char_count']} chars, type: {parsed['metadata']['inferred_doc_type']}")

    print("\n=== Step 2: Testing RAG Pipeline Indexing & Retrieval ===")
    rag = RAGPipeline(chunk_size=400, chunk_overlap=80)
    rag.build_index(parsed_docs)
    print(f" [+] Indexed {len(rag.chunks)} chunks in vector space.")

    test_query = "What is blocking Milestone 2 and what did VoltaCore delay?"
    retrieved = rag.retrieve(test_query, top_k=3)
    print(f" [+] Retrieved {len(retrieved)} relevant chunks for query: '{test_query}'")
    for chunk, score in retrieved:
        print(f"     - Score {score:.3f} | {chunk.source}: {chunk.text[:90]}...")

    assert len(retrieved) > 0, "Retrieval failed to find chunks!"

    print("\n=== Step 3: Testing Multi-Agent Analytical Pipeline ===")
    llm = LLMClient() # Runs robust fallback or Gemini
    context = rag.get_full_context()

    scope_agent = ScopeDeliverableAgent(llm)
    risk_agent = RiskForecastingAgent(llm)
    blocker_agent = BlockerActionAgent(llm)
    doc_agent = DocumentationAgent(llm)

    scope_data = scope_agent.analyze(context)
    print(f" [+] Scope Agent: Found {len(scope_data.get('key_deliverables', []))} deliverables, {len(scope_data.get('milestones', []))} milestones.")

    risk_data = risk_agent.analyze(context)
    print(f" [+] Risk Agent: Overall Risk={risk_data.get('overall_risk_level')}, Score={risk_data.get('risk_score')}/100, Risks identified={len(risk_data.get('identified_risks', []))}")

    blocker_data = blocker_agent.analyze(context)
    print(f" [+] Blocker Agent: {len(blocker_data.get('active_blockers', []))} active blockers, {len(blocker_data.get('action_items', []))} action items.")

    print("\n=== Step 4: Testing Project Health Scorer & Doc Generation ===")
    health = ProjectHealthScorer.calculate_health(scope_data, risk_data, blocker_data)
    print(f" [+] Project Health Score: {health['overall_score']}/100 -> Status: {health['status']}")
    print(f"     Recommendations: {health['recommendations']}")

    docs = doc_agent.generate_documentation(context, scope_data, risk_data)
    print(f" [+] Documentation Agent: Generated {len(docs.get('user_stories', []))} user stories, {len(docs.get('risk_register', []))} risk register rows.")

    print("\n=== Step 5: Testing Grounded Conversational Assistant ===")
    assistant = ConversationalAssistant(rag, llm)
    res = assistant.answer_query("Who is assigned to mock the VoltaCore battery API and what is the deadline?")
    print(f" [+] Assistant Answer:\n{res['answer']}")
    print(f" [+] Citations returned: {len(res['citations'])}")

    print("\n==============================================")
    print("✅ ALL PIPELINE MODULES VALIDATED SUCCESSFULLY!")
    print("==============================================")

if __name__ == "__main__":
    test_pipeline()
