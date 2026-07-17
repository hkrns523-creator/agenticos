"""Streamlit front end. Run with: streamlit run src/agenticos/ui/streamlit_app.py"""
from __future__ import annotations

import streamlit as st

from agenticos.agents.graph import build_graph
from agenticos.agents.state import new_initial_state
from agenticos.logging_config import configure_logging
from agenticos.memory.store import ConversationStore, new_conversation_id
from agenticos.settings import get_settings

configure_logging()
settings = get_settings()

st.set_page_config(page_title="AgenticOS", layout="wide", page_icon="🤖")

st.title("🤖 AgenticOS — Multi-Agent Operations Intelligence Platform")
st.caption("Planner → Specialist Data Fetchers → Supervisor, powered by a local open-source LLM")


@st.cache_resource
def get_app():
    return build_graph()


@st.cache_resource
def get_memory_store():
    if not settings.memory_enabled:
        return None
    try:
        return ConversationStore(settings.resolved_memory_db_path)
    except Exception:
        return None


if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = new_conversation_id()
if "turns" not in st.session_state:
    st.session_state.turns = []  # this-session display history: [{request, final_answer}]

AGENT_DISPLAY_NAMES = {
    "planner": "Planner",
    "run_specialists": "Specialist Data Fetch",
    "alarm_agent": "Alarm Agent",
    "asset_agent": "Asset Agent",
    "documentation_agent": "Documentation Agent",
    "energy_agent": "Energy Agent",
    "supervisor": "Supervisor",
    "no_data": "No Agents Selected",
}

with st.sidebar:
    st.header("Submit a Request")
    user_request = st.text_area(
        "Describe the issue to investigate:",
        "Investigate why AHU-01 is showing a high supply air temperature alarm and suggest next steps.",
        height=120,
    )
    submit = st.button("🚀 Run Investigation", use_container_width=True)

    st.divider()
    st.caption(f"Conversation ID: `{st.session_state.conversation_id[:12]}…`")
    st.caption("Follow-up requests (e.g. \"what about its alarms?\") reuse this conversation's memory.")
    if st.button("🆕 Start New Conversation", use_container_width=True):
        st.session_state.conversation_id = new_conversation_id()
        st.session_state.turns = []
        st.rerun()

if submit and user_request.strip():
    app = get_app()
    memory_store = get_memory_store()

    history = []
    if memory_store and settings.memory_enabled:
        history = memory_store.get_recent_turns(st.session_state.conversation_id, limit=settings.memory_max_turns)

    with st.spinner("Planning, fetching data, and synthesizing an answer..."):
        result = app.invoke(
            new_initial_state(user_request, conversation_id=st.session_state.conversation_id, history=history)
        )

    st.session_state.turns.append({"request": user_request, "final_answer": result["final_answer"]})

    chosen_agents = [a["agent"] for a in result.get("assignments", [])]
    lifecycle_steps = ["planner"] + (chosen_agents + ["supervisor"] if chosen_agents else ["no_data"])

    st.subheader("🔄 Agent Execution Flow")
    cols = st.columns(len(lifecycle_steps))
    for i, step in enumerate(lifecycle_steps):
        with cols[i]:
            elapsed = result["timings"].get(step)
            label = AGENT_DISPLAY_NAMES.get(step, step)
            if elapsed is not None:
                st.success(f"✅ {label}\n\n{elapsed}s")
            else:
                st.info(f"⏳ {label}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🧠 Agents Involved")
        for agent_name in chosen_agents:
            if agent_name in result.get("agent_errors", {}):
                st.error(f"⚠️ {AGENT_DISPLAY_NAMES.get(agent_name, agent_name)} — {result['agent_errors'][agent_name]}")
            else:
                st.success(f"✅ {AGENT_DISPLAY_NAMES.get(agent_name, agent_name)}")

        st.subheader("📜 Reasoning Log")
        for entry in result["logs"]:
            st.text(entry)

        st.subheader("📊 Token & Time Monitor")
        usage = result["token_usage"]
        total_time = sum(result["timings"].values())
        t1, t2 = st.columns(2)
        with t1:
            st.metric("Total Tokens", usage.get("total_tokens", 0))
            st.metric("Prompt Tokens", usage.get("input_tokens", 0))
        with t2:
            st.metric("Total Time", f"{total_time:.2f}s")
            st.metric("Completion Tokens", usage.get("output_tokens", 0))
        st.caption(
            "Only 2 LLM calls happen per request regardless of how many agents run "
            "(planner + supervisor) — specialist data fetches are direct DB/RAG lookups."
        )
        if usage.get("total_tokens", 0) == 0:
            st.caption("Note: the local Ollama model used here doesn't report token counts in this setup.")

    with col2:
        st.subheader("📋 Final Answer")
        st.markdown(result["final_answer"])

elif st.session_state.turns:
    st.info("👈 Enter a follow-up request in the sidebar, or review this conversation below.")
else:
    st.info("👈 Enter a request in the sidebar and click **Run Investigation** to begin.")

if st.session_state.turns:
    st.divider()
    st.subheader("🗂️ Conversation History")
    for turn in reversed(st.session_state.turns[:-1] if submit else st.session_state.turns):
        with st.expander(turn["request"]):
            st.markdown(turn["final_answer"])
