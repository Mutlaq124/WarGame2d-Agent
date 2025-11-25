"""HTTP API entrypoint for driving the game from a web UI."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.scenario import Scenario
from game_runner import GameRunner

app = FastAPI()
runner: GameRunner | None = None


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
    return runner.get_initial_frame().to_dict()


@app.post("/step")
def step(request: StepRequest):
    if runner is None:
        raise HTTPException(400, "No active game")
    if runner.done:
        raise HTTPException(400, "Game finished")
    return runner.step(request.injections).to_dict()


@app.get("/status")
def status():
    if runner is None:
        return {"active": False}
    return {"active": True, "turn": runner.turn, "step": runner.step_count, "done": runner.done}
