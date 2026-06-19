"""
Technical Analyst
-----------------

Price action, momentum indicators, and entry/exit timing.
Tools: YFinance.
"""

from agno.agent import Agent
from agno.learn import LearningMachine
from agno.models.anthropic import Claude

from agents.settings import team_knowledge, team_learnings
from agents.tracing import make_search_knowledge, setup_traced_learning, traced_yfinance_tools
from context import COMMITTEE_CONTEXT
from db import get_postgres_db

agent_db = get_postgres_db()

instructions = f"""\
You are the Technical Analyst on a $10M investment team.

## Committee Rules (ALWAYS FOLLOW)

{COMMITTEE_CONTEXT}

## Your Role

You analyze price action, momentum, and timing to determine optimal entry and
exit points for investments.

### What You Do

- Analyze price trends (50-day and 200-day moving averages)
- Evaluate momentum indicators (RSI, MACD signals)
- Identify support and resistance levels
- Assess volume patterns and breakout potential
- Provide a technical signal: **Bullish** / **Neutral** / **Bearish**

## Workflow

1. Always search learnings before analysis for relevant patterns and past insights.
2. Use YFinance for historical prices and technical indicators.
3. Identify key support/resistance levels and trend direction.
4. Save any new patterns, corrections, or insights as learnings.
5. Provide your assessment with a clear technical signal.
"""

learned_store, search_learnings, save_learning = setup_traced_learning(team_learnings)
search_knowledge = make_search_knowledge(team_knowledge)

technical_analyst = Agent(
    id="technical-analyst",
    name="Technical Analyst",
    model=Claude(id="claude-sonnet-4-6"),
    db=agent_db,
    instructions=instructions,
    tools=[traced_yfinance_tools(), search_knowledge, search_learnings, save_learning],
    knowledge=team_knowledge,
    search_knowledge=False,
    learning=LearningMachine(
        knowledge=team_learnings,
        learned_knowledge=learned_store,
    ),
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
    enable_agentic_memory=True,
)

if __name__ == "__main__":
    technical_analyst.print_response("Is NVDA in a bullish or bearish trend?", stream=True)
