"""
Agentic Investment Team
-----------------------------

A multi-agent investment team demonstrating 5 architectures.

Run:
    python -m app.main
"""

import os
from pathlib import Path
from traceloop.sdk import Traceloop
from agno.os import AgentOS

from agents import (
    committee_chair,
    financial_analyst,
    knowledge_agent,
    market_analyst,
    memo_writer,
    risk_officer,
    technical_analyst,
)
from agents.tracing import instrument_agent_entrypoints
from db import get_postgres_db
from teams import broadcast_team, coordinate_team, route_team, task_team
from workflows import investment_workflow

# ---------------------------------------------------------------------------
# Create AgentOS
# ---------------------------------------------------------------------------
agent_os = AgentOS(
    name="Agentic Investment Team",
    tracing=True,
    scheduler=True,
    db=get_postgres_db(),
    agents=[
        market_analyst,
        financial_analyst,
        technical_analyst,
        risk_officer,
        knowledge_agent,
        memo_writer,
        committee_chair,
    ],
    teams=[coordinate_team, route_team, broadcast_team, task_team],
    workflows=[investment_workflow],
    config=str(Path(__file__).parent / "config.yaml"),
)

instrument_agent_entrypoints(
    [
        (market_analyst, "Market Analyst"),
        (financial_analyst, "Financial Analyst"),
        (technical_analyst, "Technical Analyst"),
        (risk_officer, "Risk Officer"),
        (knowledge_agent, "Knowledge Agent"),
        (memo_writer, "Memo Writer"),
        (committee_chair, "Committee Chair"),
    ]
)

app = agent_os.get_app()


Traceloop.init(app_name="agno_agent", api_endpoint="http://api.overmind-dev.orb.local:8000/api", api_key="ovr_6LvPr3GRa_e8hOMtyblENFQImuLBZ5J9zBQnN_mfIRc")

# init(
#     overmind_api_key="ovr_6LvPr3GRa_e8hOMtyblENFQImuLBZ5J9zBQnN_mfIRc",
#     overmind_base_url="http://api.overmind-dev.orb.local:8000",
#     service_name="my-new-service",
#     environment="local",
#     providers=["google", "agno", "anthropic"],
# )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RUNTIME_ENV", "") == "dev"
    agent_os.serve(app="app.main:app", port=port, reload=reload)
