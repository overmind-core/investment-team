"""
Knowledge Agent
---------------

Team librarian with two retrieval modes:
- Research Library (vector search / RAG) for company and sector research
- Memo Archive (file navigation) for past investment memos
"""

import json

from agno.agent import Agent
from agno.models.anthropic import Claude

from agents.settings import MEMOS_DIR, team_knowledge
from agents.tracing import TracedFileTools, make_search_knowledge, tracer
from context import COMMITTEE_CONTEXT
from db import get_postgres_db

agent_db = get_postgres_db()
_model_id = "claude-sonnet-4-6"

instructions_prefix = f"""\
You are the Knowledge Agent on a $10M investment team. You serve as the
team's librarian with two retrieval capabilities.

## Committee Rules (ALWAYS FOLLOW)

{COMMITTEE_CONTEXT}

## Your Role

You have two retrieval modes:

"""


def research_library_rag(query: str) -> str:
    """Research Library (RAG) mode - search the knowledge base."""
    with tracer.start_as_current_span("Research Library (RAG)") as span:
        span.set_attribute("input", query)
        span.set_attribute("model", _model_id)
        try:
            docs = team_knowledge.search(query=query)
            if not docs:
                result = "No documents found"
            else:
                result = json.dumps([doc.to_dict() for doc in docs], default=str)
            span.set_attribute("output", result[:8000])
            return result
        except Exception as exc:
            span.record_exception(exc)
            raise


instructions_mode_a = """\
### Mode A — Research Library (Vector Search / RAG)
When asked about companies or sectors, search the knowledge base automatically.
This contains company research profiles and sector analysis documents loaded
via PgVector hybrid search. Good for questions like:
- "What does our research say about NVDA's competitive moat?"
- "What's the outlook for the AI semiconductor sector?"

"""


def memo_archive_file_navigation(directory: str = ".") -> str:
    """Memo Archive (File Navigation) mode - list memo files."""
    with tracer.start_as_current_span("Memo Archive (File Navigation)") as span:
        span.set_attribute("input", directory)
        span.set_attribute("model", _model_id)
        try:
            file_tools = TracedFileTools(
                base_dir=MEMOS_DIR,
                enable_read_file=True,
                enable_list_files=True,
                enable_search_files=True,
                enable_save_file=False,
                enable_delete_file=False,
            )
            result = file_tools.list_files()
            span.set_attribute("output", result[:8000])
            return result
        except Exception as exc:
            span.record_exception(exc)
            raise


instructions_mode_b = """\
### Mode B — Memo Archive (File Navigation)
When asked about past memos or historical decisions, use FileTools to list,
search, and read memo files. Memos are structured documents that should be
read in full — never summarize from fragments. Good for questions like:
- "Pull up our last NVDA memo"
- "What did we decide about TSLA last quarter?"
- "What past memos do we have on file?"

## Guidelines

- For company/sector questions: rely on the automatic knowledge base search
- For past memos/decisions: use list_files, search_files, and read_file
- Always read memos completely — never summarize from fragments
- Provide specific citations with filenames and dates
- Surface relevant historical precedents when they exist
- If information isn't available, say so clearly
"""

instructions = instructions_prefix + instructions_mode_a + instructions_mode_b
search_knowledge = make_search_knowledge(team_knowledge)

knowledge_agent = Agent(
    id="knowledge-agent",
    name="Knowledge Agent",
    model=Claude(id=_model_id),
    db=agent_db,
    instructions=instructions,
    tools=[
        TracedFileTools(
            base_dir=MEMOS_DIR,
            enable_read_file=True,
            enable_list_files=True,
            enable_search_files=True,
            enable_save_file=False,
            enable_delete_file=False,
        ),
        research_library_rag,
        memo_archive_file_navigation,
        search_knowledge,
    ],
    knowledge=team_knowledge,
    search_knowledge=False,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
    enable_agentic_memory=True,
)
