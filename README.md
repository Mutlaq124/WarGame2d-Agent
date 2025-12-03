# Grid Combat Environment (Developer Quickstart)

2D turn-based air combat sim with a FastAPI backend and a lightweight HTML control panel.

## Get Running (Python already installed)
1) Install uv (once): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2) Install deps: `uv sync`
3) Launch backend + UI: `uv run python main.py`
   - Starts FastAPI (`api/app.py`) and serves `ui/ops_deck.html`
   - Auto-opens your browser; logs go to stdout and `storage/logs/backend.log`

## What You’re Running
- API: `/start`, `/step`, `/status` endpoints driving the simulation
- UI: polls the API to visualize turns and send steps
- Engine: `env/` holds the simulation, `runtime/runner.py` orchestrates games

## Scenarios
- Scenarios define grid, rules, entities, and agents (`env/scenario.py`).
- Use the API or code to load/run them; recordings can be produced via the API flow.

## Create a New Agent
```python
# agents/my_agent.py
from agents import BaseAgent, register_agent
from env.core.types import Team

@register_agent("my_agent")
class MyAgent(BaseAgent):
    def __init__(self, team: Team, **kwargs):
        super().__init__(team, name="MyAgent")
    def get_actions(self, state, step_info=None, **kwargs):
        return {}, {"policy": "my_agent"}
```
- Registry auto-discovers agent modules on first use; no manual imports needed.
- Use in a scenario via `AgentSpec(type="my_agent", team=Team.BLUE, init_params={...})`.

## Logging
- Configured once in `main.py` via `configure_logging()`; per-module loggers via `get_logger(__name__)`.
- Defaults to console + `storage/logs/backend.log` with line numbers.

## Need More Structure?
See `ARCHITECTURE.md` for the module map. Keep changes simple: run `uv sync`, run `uv run python main.py`, add agents via `@register_agent`.
