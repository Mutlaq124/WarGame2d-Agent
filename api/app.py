"""HTTP API entrypoint for driving the game from a web UI."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.scenario import Scenario
from game_runner import GameRunner

app = FastAPI()
runner: GameRunner | None = None

# Allow the browser-based control panel (served from file:// or other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    scenario: dict
    world: dict | None = None

class StepRequest(BaseModel):
    injections: dict | None = None


@app.post("/start")
def start(request: StartRequest):
    global runner
    scenario = Scenario.from_dict(request.scenario)
    runner = GameRunner(scenario, world=request.world)
    return {"success": True}


@app.post("/step")
def step(request: StepRequest):
    if runner is None:
        raise HTTPException(400, "No active game")
    try:
        return runner.step(request.injections).to_dict()
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/status")
def status():
    if runner is None:
        return {"active": False}
    return {"active": True, "turn": runner.turn, "step": runner.step_count, "done": runner.done}
