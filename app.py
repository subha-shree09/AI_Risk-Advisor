import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

from src.ingestion import DocumentIngestionEngine
from src.rag_engine import RAGPipeline
from src.llm_client import LLMClient
from src.agents.scope_agent import ScopeDeliverableAgent
from src.agents.risk_agent import RiskForecastingAgent
from src.agents.blocker_agent import BlockerActionAgent
from src.agents.documentation_agent import DocumentationAgent
from src.health_scorer import ProjectHealthScorer
from src.conversational_assistant import ConversationalAssistant

st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
    }
    .risk-badge-low { background-color: #DEF7EC; color: #03543F; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .risk-badge-medium { background-color: #FEF08A; color: #713F12; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .risk-badge-high { background-color: #FED7AA; color: #9A3412; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
    .risk-badge-critical { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 10px; border-radius: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar API Key & Settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Project Advisor")
    st.caption("AI-Powered Risk & Delivery Intelligence")
    
    st.divider()
    api_key_input = st.text_input(
        "Gemini API Key (Optional)",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Optional: Enter a Gemini API key for live neural LLM inferences. If omitted, the system seamlessly operates on its deterministic offline heuristic agent engine."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input

    st.markdown("### 📊 Ingested Knowledge Base")
    if st.session_state.ingested_docs:
        st.success(f"📂 {len(st.session_state.ingested_docs)} Document(s) Indexed")
        for doc in st.session_state.ingested_docs:
            st.markdown(f"- **{doc['metadata']['filename']}** (`{doc['metadata']['inferred_doc_type']}`)")
    else:
        st.info("No documents uploaded yet.")

    st.divider()
    st.markdown("💡 **Milestones Covered:**")
    st.markdown("✅ M1: RAG & Multi-Format Ingestion\n✅ M2: Multi-Agent Intelligence\n✅ M2: Health Scoring & User Stories")

# Top Header
st.markdown('<div class="main-header">🛡️ AI Project Intelligence & Risk Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unified Document Ingestion, Multi-Agent Analytics, Delivery Forecasting & Grounded Assistant</div>', unsafe_allow_html=True)

# Tabs
tab_ingest, tab_insights, tab_risks, tab_docs, tab_assistant = st.tabs([
    "📥 1. Document Ingestion & RAG",
    "📈 2. Insights & Health Score",
    "⚠️ 3. Risk & Blocker Matrix",
    "📝 4. Documentation Generator",
    "💬 5. Grounded Project Assistant"
])

# Initialize LLM Client
llm_client = LLMClient(api_key=api_key_input if api_key_input else None)

# ================= TAB 1: INGESTION =================
with tab_ingest:
    st.subheader("Upload Project Artifacts")
    st.write("Upload any combination of project documents — Proposals (PDF), SRS (DOCX), Task Lists (CSV), Sprint Notes (TXT/MD).")

    uploaded_files = st.file_uploader(
        "Choose project files", 
        type=["pdf", "docx", "csv", "txt", "md"], 
        accept_multiple_files=True
    )

    col_btn1, col_btn2 = st.columns([2, 5])
    with col_btn1:
        process_btn = st.button("🚀 Process & Build Knowledge Base", use_container_width=True, type="primary")

    if process_btn and uploaded_files:
        with st.spinner("Parsing documents and building RAG vector index..."):
            parsed_docs = []
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                try:
                    res = DocumentIngestionEngine.parse_file(uploaded_file.name, file_bytes)
                    parsed_docs.append(res)
                except Exception as e:
                    st.error(f"Error parsing {uploaded_file.name}: {e}")

            if parsed_docs:
                st.session_state.ingested_docs = parsed_docs
                st.session_state.rag_pipeline.build_index(parsed_docs)
                st.success(f"Successfully processed {len(parsed_docs)} files and created {len(st.session_state.rag_pipeline.chunks)} semantic chunks in vector knowledge base!")

                # Automatically trigger multi-agent pipeline
                with st.spinner("Running Multi-Agent analytical pipeline (Scope, Risk, Blocker, Health)..."):
                    context_text = st.session_state.rag_pipeline.get_full_context()
                    
                    scope_agent = ScopeDeliverableAgent(llm_client)
                    risk_agent = RiskForecastingAgent(llm_client)
                    blocker_agent = BlockerActionAgent(llm_client)
                    doc_agent = DocumentationAgent(llm_client)

                    scope_res = scope_agent.analyze(context_text)
                    risk_res = risk_agent.analyze(context_text)
                    blocker_res = blocker_agent.analyze(context_text)
                    health_res = ProjectHealthScorer.calculate_health(scope_res, risk_res, blocker_res)
                    doc_res = doc_agent.generate_documentation(context_text, scope_res, risk_res)

                    st.session_state.analysis_results = {
                        "scope": scope_res,
                        "risk": risk_res,
                        "blocker": blocker_res,
                        "health": health_res,
                        "docs": doc_res
                    }
                st.rerun()

    # Ingestion Status Table
    if st.session_state.ingested_docs:
        st.markdown("### Processed Artifacts Summary")
        summary_data = []
        for d in st.session_state.ingested_docs:
            meta = d['metadata']
            summary_data.append({
                "Filename": meta['filename'],
                "Format": meta['file_type'].upper(),
                "Inferred Role": meta['inferred_doc_type'],
                "Characters": f"{meta['char_count']:,}",
                "Lines": meta['line_count']
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ================= TAB 2: INSIGHTS & HEALTH =================
with tab_insights:
    if not st.session_state.analysis_results:
        st.warning("Please upload and process documents in Tab 1 to generate project intelligence.")
    else:
        results = st.session_state.analysis_results
        health = results["health"]
        scope = results["scope"]
        risk = results["risk"]

        # Health Hero Banner
        st.subheader("Project Health & Executive Scorecard")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(
                label="Overall Health Score", 
                value=f"{health['overall_score']} / 100", 
                delta=f"Status: {health['status']}",
                delta_color="normal" if health['overall_score'] >= 70 else "inverse"
            )
        with m_col2:
            st.metric(
                label="Overall Risk Score", 
                value=f"{risk.get('risk_score', 0)} / 100",
                delta=f"Level: {risk.get('overall_risk_level', 'MED')}",
                delta_color="inverse"
            )
        with m_col3:
            st.metric(
                label="Schedule Confidence", 
                value=risk.get("schedule_forecast", {}).get("estimated_completion_confidence", "N/A"),
                delta="Delivery Forecast"
            )
        with m_col4:
            st.metric(
                label="Active Blockers", 
                value=len(results["blocker"].get("active_blockers", [])),
                delta="Unresolved Items",
                delta_color="inverse" if len(results["blocker"].get("active_blockers", [])) > 0 else "normal"
            )

        # Health Breakdown Radar / Bar Chart
        col_chart, col_recs = st.columns([1, 1])
        with col_chart:
            st.markdown("#### Health Dimension Breakdown")
            breakdown = health["breakdown"]
            fig = go.Figure(data=go.Scatterpolar(
                r=[breakdown["scope_clarity"], breakdown["schedule_confidence"], breakdown["risk_mitigation"], breakdown["blocker_velocity"]],
                theta=['Scope Clarity', 'Schedule Confidence', 'Risk Mitigation', 'Blocker Velocity'],
                fill='toself',
                line=dict(color=health['color'])
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                margin=dict(l=40, r=40, t=20, b=20),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_recs:
            st.markdown("#### Actionable Recommendations")
            st.info(health["summary"])
            for rec in health["recommendations"]:
                st.markdown(f"👉 {rec}")

        st.divider()

        # Scope & Deliverables
        st.subheader("Scope & Deliverable Extraction")
        st.markdown(f"**Primary Objective:** {scope.get('primary_objective', 'N/A')}")
        
        st.markdown("#### Key Deliverables")
        deliv_df = pd.DataFrame(scope.get("key_deliverables", []))
        if not deliv_df.empty:
            st.dataframe(deliv_df, use_container_width=True)
        else:
            st.write("No deliverables found.")

# ================= TAB 3: RISKS & BLOCKERS =================
with tab_risks:
    if not st.session_state.analysis_results:
        st.warning("Please upload and process documents in Tab 1.")
    else:
        results = st.session_state.analysis_results
        risk_data = results["risk"]
        blocker_data = results["blocker"]

        st.subheader("Risk Detection & Delivery Forecasting")
        st.write(risk_data.get("schedule_forecast", {}).get("forecast_summary", ""))

        st.markdown("#### Identified Risks")
        risks = risk_data.get("identified_risks", [])
        if risks:
            for r in risks:
                with st.expander(f"[{r.get('category')}] {r.get('title')} (Impact: {r.get('impact')}, Likelihood: {r.get('likelihood')})"):
                    st.markdown(f"**Description:** {r.get('description')}")
                    st.markdown(f"**Mitigation Strategy:** {r.get('mitigation_strategy')}")
                    if r.get("evidence_snippet"):
                        st.caption(f"Evidence: {r.get('evidence_snippet')}")
        else:
            st.success("No critical risks detected.")

        st.divider()
        st.subheader("Blockers & Pending Decisions")
        
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.markdown("#### Active Blockers")
            blockers = blocker_data.get("active_blockers", [])
            for b in blockers:
                st.error(f"🛑 **{b.get('issue')}**\n\nImpact: {b.get('impact')} | Owner: {b.get('resolution_owner', 'Unassigned')}\n\n*Action:* {b.get('suggested_action')}")

        with b_col2:
            st.markdown("#### Pending Decisions")
            decisions = blocker_data.get("pending_decisions", [])
            for d in decisions:
                st.warning(f"⚖️ **{d.get('topic')}** (`{d.get('status')}`)\n\nOptions: {', '.join(d.get('options', []))}\n\n*Impact:* {d.get('impact')}")

        st.markdown("#### Assigned Action Items")
        actions_df = pd.DataFrame(blocker_data.get("action_items", []))
        if not actions_df.empty:
            st.dataframe(actions_df, use_container_width=True)

# ================= TAB 4: DOCUMENTATION GENERATOR =================
with tab_docs:
    if not st.session_state.analysis_results:
        st.warning("Please upload and process documents in Tab 1.")
    else:
        docs = st.session_state.analysis_results["docs"]

        st.subheader("Automated Project Documentation Generation")
        st.caption("Auto-generated User Stories, Acceptance Criteria, and formal Risk Register.")

        doc_subtab1, doc_subtab2, doc_subtab3 = st.tabs(["User Stories", "Risk Register", "Export Markdown"])
        
        with doc_subtab1:
            for us in docs.get("user_stories", []):
                st.markdown(f"### {us.get('story_id')}: {us.get('title')}")
                st.markdown(f"**As a** *{us.get('role')}*, **I want to** *{us.get('action')}*, **So that** *{us.get('benefit')}*.")
                st.markdown("**Acceptance Criteria:**")
                for ac in us.get("acceptance_criteria", []):
                    st.checkbox(ac, key=f"{us.get('story_id')}_{ac[:15]}")
                st.divider()

        with doc_subtab2:
            rr_df = pd.DataFrame(docs.get("risk_register", []))
            if not rr_df.empty:
                st.dataframe(rr_df, use_container_width=True)

        with doc_subtab3:
            st.download_button(
                label="📥 Download Documentation Package (.md)",
                data=docs.get("markdown_export", ""),
                file_name="project_documentation_package.md",
                mime="text/markdown"
            )
            st.code(docs.get("markdown_export", ""), language="markdown")

# ================= TAB 5: CONVERSATIONAL ASSISTANT =================
with tab_assistant:
    st.subheader("Conversational Project Intelligence Assistant")
    st.caption("Ask questions about deliverables, timelines, risks, and health grounded strictly in uploaded documents.")

    chat_assistant = ConversationalAssistant(st.session_state.rag_pipeline, llm_client)

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            if "citations" in chat and chat["citations"]:
                with st.expander("📚 Grounded Citations"):
                    for c in chat["citations"]:
                        st.markdown(f"- **{c['source']}** (Section {c['section']}): `{c['snippet']}`")

    # User input
    if user_query := st.chat_input("Ask about project scope, risks, blockers, or deadlines..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate grounded response
        with st.chat_message("assistant"):
            with st.spinner("Searching vector knowledge base and synthesizing grounded answer..."):
                response = chat_assistant.answer_query(user_query)
                st.markdown(response["answer"])
                if response["citations"]:
                    with st.expander("📚 Grounded Citations"):
                        for c in response["citations"]:
                            st.markdown(f"- **{c['source']}** (Section {c['section']}): `{c['snippet']}`")
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "citations": response["citations"]
                })
