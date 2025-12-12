"""
Multi-stage LLM-powered agent that uses Groq for strategic and tactical decisions.
"""

import json
import random
import re
from typing import Dict, Any, Optional, TYPE_CHECKING, List

from env.core.types import Team, ActionType
from env.core.actions import Action
from env.world import WorldState
from ..base_agent import BaseAgent
from ..team_intel import TeamIntel
from ..registry import register_agent

from .groq_client import GroqClient
from .ollama_client import OllamaClient
import os
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from env.environment import StepInfo

@register_agent("llm_basic")
class LLMAgent(BaseAgent):
    """
    Multi-stage LLM agent that:
    1. Strategic planning (overall goal)
    2. Tactical decisions (per-entity actions)
    3. Action execution (map to game actions)
    """

    def __init__(
        self,
        team: Team,
        name: str = None,
        seed: Optional[int] = None,
        provider: str = "ollama", # Choose Provider
        model: str = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_memory: Optional[int] = None,  # NEW: Allow override
        **_: Any,
    ):
        """Initialize multi-stage LLM agent."""
        super().__init__(team, name or f"LLMAgent-{team.name}")
        self.rng = random.Random(seed)
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")

        if self.provider == "groq":
            self.model = "llama-3.3-70b-versatile"
        elif self.provider == "ollama":
            self.model = "llama3.1:8b-instruct-q4_K_S"
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        self._init_llm_client()

        # NEW: Memory size adapts to scenario length
        # Default: remember last 10% of game turns, min 3, max 10
        self.max_memory = max_memory if max_memory else 5  # Can be overridden
        
        self.strategy_memory: List[Dict[str, Any]] = []
        self.current_turn = 0

    def get_actions(
        self,
        state: Dict[str, Any],
        step_info: Optional["StepInfo"] = None,
        **kwargs: Any,
    ) -> tuple[Dict[int, Action], Dict[str, Any]]:
        """
        called in every game step to get the actions for all entities controlled by this agent.

        
                        """
        world: WorldState = state["world"]
        intel: TeamIntel = TeamIntel.build(world, self.team)
        self.current_turn = world.turn
        
        # Collect allowed actions
        allowed_actions: Dict[int, List[Action]] = {}
        for entity in intel.friendlies:
            if not entity.alive:
                continue
# restrict allowed actions to only alive entities
            allowed = entity.get_allowed_actions(world)
            if allowed:
                allowed_actions[entity.id] = allowed
# No try-except, let it raise if LLM fails 
        # Stage 1: Strategic Planning
        strategy = self._get_strategy(world, intel)
        logger.debug(f"[Stage 1] Strategy: {strategy}")
        
        # Stage 2: Tactical Decisions (per entity)
        actions = self._get_tactical_actions(intel, allowed_actions, strategy, world)
        
        metadata = {
            "strategy": strategy,
            "allowed_actions": allowed_actions,
            "used_llm": True
        }
        return actions, metadata

    def _get_strategy(self, world: WorldState, intel: TeamIntel) -> Dict[str, Any]:
        """Stage 1: High-level strategic planning."""
        prompt = self._build_strategy_prompt(world, intel)
        
        system_prompt = f"""You are a tactical air combat commander for {self.team.name} team.
Analyze the battlefield and determine the overall strategy."""
        
        response = self.llm.complete(system_prompt, prompt, temperature=0.7)
        strategy = self._parse_strategy(response)
        
        # Store in memory
        self.strategy_memory.append(strategy)
        if len(self.strategy_memory) > self.max_memory:
            self.strategy_memory.pop(0)
        
        return strategy

    def _build_strategy_prompt(self, world: WorldState, intel: TeamIntel) -> str:
        """Build strategic analysis prompt."""
        # Game state summary
        state_text = f"Turn: {world.turn}\n"
        state_text += f"Grid: {world.grid.width}x{world.grid.height}\n\n"
        
        # My forces
        state_text += f"Your Forces ({self.team.name}):\n"
        for entity in intel.friendlies:
            if not entity.alive:
                continue
            state_text += f"  - {entity.kind.upper()} #{entity.id} at ({entity.pos[0]}, {entity.pos[1]})"
            if hasattr(entity, 'missiles'):
                state_text += f", Missiles: {entity.missiles}"
            state_text += "\n"
        
        # Enemy forces
        state_text += f"\nEnemy Forces:\n"
        for entity in intel.enemies:
            if not entity.alive:
                continue
            state_text += f"  - {entity.kind.upper()} #{entity.id} at ({entity.pos[0]}, {entity.pos[1]})\n"
        
        # Past strategies
        memory_text = ""
        if self.strategy_memory:
            memory_text = "\nPast Strategies:\n"
            for s in self.strategy_memory[-3:]:
                memory_text += f"  Turn {s.get('turn', '?')}: {s.get('strategic_goal', 'N/A')}\n"
        
        return f"""{state_text}
{memory_text}

Analyze the situation:
1. Who has the advantage (BLUE/RED/NEUTRAL), based on Number of units alive, Missile counts, Position?
2. What are the key threats to your team (Enemy units closer to your units, Low missile counts, poor positioning)?
3. What opportunities can you exploit(Positional advantages, missiles available, enemy weaknesses)?
4. What should be your main strategic goal this turn (Aggressive attack, Defensive hold, Balanced approach)?
Prioritize to move towards enemy targets like AWACS, Aircraft, SAM, decoys.
Respond in JSON:
{{
    "advantage": "BLUE/RED/NEUTRAL",
    "key_threats": ["threat description"],
    "opportunities": ["opportunity description"],
    "strategic_goal": "your main objective",
    "priority": "AGGRESSIVE/DEFENSIVE/BALANCED"
}}"""

    def _parse_strategy(self, response: str) -> Dict[str, Any]:
        """Parse strategic LLM response."""
        try:
            json_str = self._extract_json(response)
            strategy = json.loads(json_str)
            strategy['turn'] = self.current_turn
            return strategy
        except Exception as e:
            print(f"[{self.name}] Strategy parse failed: {e}")
            # return None # Fallback to None (don't use any fallback strategy)

    def _get_tactical_actions(
# For each friendy entity, build entity-specific prompt with strategic goal, entity status, nearby enemies(sort by distance), allowed actions
# LLM chooses action index in JSON response    
        self,
        intel: TeamIntel,
        allowed_actions: Dict[int, List[Action]],
        strategy: Dict[str, Any],
        world: WorldState
    ) -> Dict[int, Action]:
        """Stage 2: Get tactical decisions for each entity."""
        actions: Dict[int, Action] = {}
        
        for entity in intel.friendlies:
            if not entity.alive or entity.id not in allowed_actions:
                continue
            
            # Build entity-specific prompt
            prompt = self._build_entity_prompt(entity, intel, strategy, allowed_actions[entity.id], world)
            
            system_prompt = f"""You are controlling a {entity.kind.upper()} unit in tactical combat.
Choose the best action based on the strategic goal.

Response format: 
Respond only in JSON (No markdown or explanation).

"""
            
            response = self.llm.complete(system_prompt, prompt, temperature=0.5)
            action = self._parse_entity_action(response, entity.id, allowed_actions[entity.id])
            
            if action:
                actions[entity.id] = action
                logger.info(f"[Entity {entity.id}] Chose action {action.type.name}")
        
        # Fill missing with random (remove it -> no action, with log, to know and debug)
        for entity_id, allowed in allowed_actions.items():
            if entity_id not in actions:
                actions[entity_id] = self.rng.choice(allowed)
        
        return actions

    def _build_entity_prompt(
        self,
        entity: Any,
        intel: TeamIntel,
        strategy: Dict[str, Any],
        allowed: List[Action],
        world: WorldState
    ) -> str:
        """Enhanced prompt for individual entity decision-making.
-> seperate shoortable(distance) and nearby targets(llm clarity), priority includes, range info include, action formatting 
        """
        
        # Calculate shootable targets
        entity_pos = getattr(entity, 'pos', (0, 0))
        missile_range = getattr(entity, 'missile_max_range', 0)
        
        shootable_enemies = []
        nearby_enemies = []
        
        for enemy in intel.enemies:
            if not enemy.alive:
                continue
            enemy_pos = getattr(enemy, 'pos', (0, 0))
            dist = abs(entity_pos[0] - enemy_pos[0]) + abs(entity_pos[1] - enemy_pos[1])
            
            if dist <= missile_range:
                shootable_enemies.append((enemy, dist))
            elif dist <= 10:
                nearby_enemies.append((enemy, dist))
        
        # Format shootable targets (HIGH PRIORITY)
        shootable_text = "Enemies IN RANGE (can shoot now):\n"
        if shootable_enemies:
            for enemy, dist in sorted(shootable_enemies, key=lambda x: x[1]):
                kind = getattr(enemy, 'kind', 'unknown').upper()
                pos = getattr(enemy, 'pos', (0, 0))
                
                # Priority based on entity type
                if kind == "AWACS":
                    priority = "HIGH"
                elif kind == "AIRCRAFT":
                    priority = "MEDIUM"
                elif kind == "SAM":
                    priority = "MEDIUM"
                elif kind == "DECOY":
                    priority = "LOW"
                else:
                    priority = "LOW"
                
                shootable_text += f"  ✓ {kind} #{enemy.id} at ({pos[0]}, {pos[1]}), distance {dist} [Priority: {priority}]\n"
        else:
            shootable_text += "  None (all enemies out of range)\n"
         
        # Format nearby but NOT shootable
        nearby_text = "Enemies NEARBY but out of range:\n"
        if nearby_enemies:
            for enemy, dist in sorted(nearby_enemies, key=lambda x: x[1])[:3]:
                kind = getattr(enemy, 'kind', 'unknown').upper()
                pos = getattr(enemy, 'pos', (0, 0))
                nearby_text += f"  - {kind} #{enemy.id} at ({pos[0]}, {pos[1]}), distance {dist}\n"
        else:
            nearby_text += "  None\n"
        
        # Format entity status
        status = f"""Entity: {entity.kind.upper()} #{entity.id}
Position: ({entity.pos[0]}, {entity.pos[1]})
"""
        if hasattr(entity, 'missiles'):
            status += f"Missiles: {entity.missiles}\n"
        
        # Format allowed actions
        action_text = "Allowed Actions:\n"
        for idx, action in enumerate(allowed):
            action_text += f"  {idx}: {action.type.name}"
            if action.target_id:
                action_text += f" -> Entity #{action.target_id}"
            elif action.target_pos:
                action_text += f" -> ({action.target_pos[0]}, {action.target_pos[1]})"
            action_text += "\n"
        
        return f"""Strategic Goal: {strategy.get('strategic_goal')}
Priority: {strategy.get('priority')}

YOUR UNIT STATUS:
  Type: {getattr(entity, 'kind', 'unknown').upper()} #{entity.id}
  Position: ({entity_pos[0]}, {entity_pos[1]})
  Missiles: {getattr(entity, 'missiles', 'N/A')}
  Missile Range: {missile_range} steps (Manhattan distance)

{shootable_text}
{nearby_text}

ALLOWED ACTIONS:
{self._format_actions(allowed)}

DECISION RULES:
1. If enemies IN RANGE: Prioritize AWACS > Aircraft > SAM > Decoy
2. If no enemies in range: Move closer OR hold position (based on strategy)
3. Conserve missiles: Don't shoot at low-priority targets if low on ammo
4. Defensive strategy: Protect valuable units (AWACS, SAM)

Choose action index (0-{len(allowed)-1}) in JSON:
{{"action_index": 0, "reasoning": "why"}}"""

    def _format_actions(self, allowed: List[Action]) -> str:
        """Format actions with distance info."""
        lines = []
        for idx, action in enumerate(allowed):
            line = f"  {idx}: {action.type.name}"
            if action.target_id:
                line += f" at Entity #{action.target_id}"
            elif action.target_pos:
                line += f" to ({action.target_pos[0]}, {action.target_pos[1]})"
            lines.append(line)
        return "\n".join(lines)

    def _parse_entity_action(
        self,
        response: str,
        entity_id: int,
        allowed: List[Action]
    ) -> Optional[Action]:
        """Parse entity action from LLM response."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            idx = data.get("action_index", 0)
            
            if 0 <= idx < len(allowed):
                return allowed[idx]
            return allowed[0]  # Fallback to first action
            
        except Exception as e:
            print(f"[{self.name}] Action parse failed for entity {entity_id}: {e}")
            return allowed[0] if allowed else None

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        import re
        
        # Try ```json ... ```
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try ``` ... ```
        match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try any JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        
        raise ValueError("No JSON found")