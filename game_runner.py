from __future__ import annotations

from typing import Any, Dict, Optional

from agents import PreparedAgent, create_agent_from_spec
from env import GridCombatEnv
from env.core.types import Team
from env.environment import StepInfo
from env.scenario import Scenario
from env.world import WorldState

from game_frame import Frame


class GameRunner:
    """
    Step-by-step game runner that returns UI-friendly frames.

    Use get_initial_frame() before any actions, then step() until done.
    """

    def __init__(
        self,
        scenario: Scenario,
        world: WorldState | Dict[str, Any] | None = None,
        verbose: bool = False,
    ):
        self.scenario = scenario.clone()
        self.verbose = verbose

        self.env = GridCombatEnv(verbose=verbose)
        self._state = self.env.reset(scenario=self.scenario, world=world)

        self._blue_agent = self._agent_from_scenario(self.scenario, Team.BLUE)
        self._red_agent = self._agent_from_scenario(self.scenario, Team.RED)

        self._done = False
        self._last_info: StepInfo | None = None
        self._final_world: WorldState | None = None

    # ------------------------------------------------------------------#
    # Properties
    # ------------------------------------------------------------------#
    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    @property
    def done(self) -> bool:
        return self._done

    @property
    def turn(self) -> int:
        """Current turn pulled directly from the world state."""
        world: WorldState = self._state["world"]
        if world is None:
            raise RuntimeError("World state is not initialized")
        return world.turn

    @property
    def step_count(self) -> int:
        """Alias for turn (backwards compatibility)."""
        return self.turn

    # ------------------------------------------------------------------#
    # Core API
    # ------------------------------------------------------------------#
    def get_initial_frame(
        self,
    ) -> Frame:
        """
        Return the frame for step 0 before any agent actions run.

        Actions, action_metadata, and info are omitted because no step
        has been taken yet.
        """
        return Frame(
            world=self._state["world"],
        )

    def step(
        self,
        injections: Optional[Dict[str, Any]] = None,
    ) -> Frame:
        """
        Execute one turn of the game and return a formatted frame.

        Args:
            injections: Optional dict with 'blue'/'red' keys for agent kwargs.
        """
        if self._done:
            if self._final_world is not None:
                final_world = self._final_world
                self._final_world = None
                return Frame(world=final_world, done=True)
            raise RuntimeError("Game is already finished")

        injections = injections or {}
        world_before: WorldState = self._state["world"].clone()
        blue_actions, blue_meta = self._blue_agent.agent.get_actions(
            self._state,
            step_info=self._last_info,
            **injections.get("blue", {}),
        )
        red_actions, red_meta = self._red_agent.agent.get_actions(
            self._state,
            step_info=self._last_info,
            **injections.get("red", {}),
        )

        merged_actions = {**blue_actions, **red_actions}
        self._state, _rewards, self._done, self._last_info = self.env.step(merged_actions)

        if self._done:
            self._final_world = self._state["world"].clone()

        return Frame(
            world=world_before,
            actions=merged_actions,
            action_metadata={"blue": blue_meta, "red": red_meta},
            step_info=self._last_info,
            done=self._done,
        )

    def run(self, *, include_history: bool = False) -> Frame | list[Frame]:
        """
        Run the full episode to completion.

        Returns the final frame, or the full frame history if include_history
        is True.
        """
        frames: list[Frame] = [self.get_initial_frame()]
        while not self._done:
            frames.append(self.step())

        return frames if include_history else frames[-1]

    def run_episode(self) -> Frame:
        """Backward-compatible alias for running to completion."""
        return self.run()  # type: ignore[return-value]


    def get_final_frame(self) -> Frame:
        """
        Return the final world state without actions for terminal view.
        """
        world: WorldState | None = self._state.get("world")
        return Frame(world=world.clone() if world else None, done=True)

    # Helpers
    def _agent_from_scenario(self, scenario: Scenario, team: Team) -> PreparedAgent:
        if not scenario.agents:
            raise ValueError("Scenario is missing agent specs.")
        matches = [spec for spec in scenario.agents if spec.team == team]
        if not matches:
            raise ValueError(f"No AgentSpec found for team {team}")
        if len(matches) > 1:
            raise ValueError(f"Multiple AgentSpecs found for team {team}")
        return create_agent_from_spec(matches[0])
