import json
import random
import re
from typing import Dict, Any, Optional, TYPE_CHECKING, List
 
from env.core.types import Team, ActionType, EntityKind
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
from dotenv import load_dotenv
load_dotenv()

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
        provider: str = "groq",  # CHANGE: Default to groq
        model: str = None,
        api_key: Optional[str] = None,
        max_memory: Optional[int] = None,
        **_: Any,
    ):
        """Initialize multi-stage LLM agent."""
        super().__init__(team, name or f"LLMAgent-{team.name}")
        self.rng = random.Random(seed)
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")  # CHANGE: Use GROQ_API_KEY

        if self.provider == "groq":
            self.model = model or "llama-3.1-8b-instant"  
        elif self.provider == "ollama":
            self.model = "llama3.1:8b-instruct-q4_K_S"
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        self._init_llm_client()

        self.max_memory = max_memory if max_memory else 5  # Can be overridden
        
        self.strategy_memory: List[Dict[str, Any]] = []
        self.current_turn = 0
        self.last_action_reasonings: Dict[int, str] = {}

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
            "action_reasonings": getattr(self, 'last_action_reasonings', {}),
            "used_llm": True
        }
        return actions, metadata

    def _get_strategy(self, world: WorldState, intel: TeamIntel) -> Dict[str, Any]:
        """Stage 1: High-level strategic planning."""
        prompt = self._build_strategy_prompt(world, intel)
        
        system_prompt = f"""You are a tactical air combat commander for {self.team.name} team.
Analyze the battlefield and determine the overall strategy."""
        
        response = self.llm.complete(system_prompt, prompt, temperature=0.2)
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
            kind_str = entity.kind.name if isinstance(entity.kind, EntityKind) else str(entity.kind).upper()
            state_text += f"  - {kind_str} #{entity.id} at ({entity.pos[0]}, {entity.pos[1]})"
            if hasattr(entity, 'missiles'):
                state_text += f", Missiles: {entity.missiles}"
            state_text += "\n"
        
        # Enemy forces - FIXED: Use visible_enemies instead of enemies
        state_text += f"\nEnemy Forces:\n"
        for enemy in intel.visible_enemies:
            kind_str = enemy.kind.name if isinstance(enemy.kind, EntityKind) else str(enemy.kind).upper()
            state_text += f"  - {kind_str} #{enemy.id} at ({enemy.position[0]}, {enemy.position[1]})\n"
        
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
4. What should be your main strategic goal this turn?
Prioritize to move towards enemy targets like AWACS, Aircraft, SAM, decoys.
Respond in JSON:
{{
    "advantage": "BLUE/RED/NEUTRAL",
    "key_threats": ["threat description"],
    "opportunities": ["opportunity description"],
    "strategic_goal": "your main objective"
}}"""

    def _parse_strategy(self, response: str) -> Dict[str, Any]:
        """Parse strategic LLM response."""
        try:
            json_str = self._extract_json(response)
            strategy = json.loads(json_str)
            strategy['turn'] = self.current_turn
            return strategy
        except Exception as e:
            logger.error(f"[{self.name}] Strategy parse failed: {e}\nResponse: {response[:200]}")
            # Return fallback strategy
            return {
                "turn": self.current_turn,
                "advantage": "NEUTRAL",
                "key_threats": ["Parse error"],
                "opportunities": [],
                "strategic_goal": "Defensive hold due to parse error"
            }

    def _get_tactical_actions(
        self,
        intel: TeamIntel,
        allowed_actions: Dict[int, List[Action]],
        strategy: Dict[str, Any],
        world: WorldState
    ) -> Dict[int, Action]:
        """Stage 2: Get tactical decisions for ALL entities in one LLM call."""
        
        # Build structured data for all entities
        units_data = []
        id_order = []
        
        for entity in intel.friendlies:
            if not entity.alive or entity.id not in allowed_actions:
                continue
            
            id_order.append(entity.id)
            
            # Calculate enemies in range
            entity_pos = entity.pos
            missile_range = getattr(entity, 'missile_max_range', 0)
            
            enemies_in_range = []
            for enemy in intel.visible_enemies:
                enemy_pos = enemy.position
                dist = abs(entity_pos[0] - enemy_pos[0]) + abs(entity_pos[1] - enemy_pos[1])
                
                if dist <= missile_range:
                    kind_str = enemy.kind.name if isinstance(enemy.kind, EntityKind) else str(enemy.kind).upper()
                    enemies_in_range.append({
                        "id": enemy.id,
                        "type": kind_str,
                        "distance": dist
                    })
            
            # Format allowed actions - OPTIMIZED WITH DISTANCE CONTEXT
            allowed_formatted = []
            for idx, action in enumerate(allowed_actions[entity.id]):
                action_data = {
                    "index": idx,
                    "type": action.type.name
                }
                
                # SHOOT action: extract target_id from params
                if action.type == ActionType.SHOOT and "target_id" in action.params:
                    action_data["target_id"] = action.params["target_id"]
                    
                    # ENHANCEMENT: Add target type for better LLM reasoning
                    target_id = action.params["target_id"]
                    for enemy in intel.visible_enemies:
                        if enemy.id == target_id:
                            kind_str = enemy.kind.name if isinstance(enemy.kind, EntityKind) else str(enemy.kind).upper()
                            action_data["target_type"] = kind_str
                            break

                # MOVE action: extract direction and calculate tactical value
                elif action.type == ActionType.MOVE and "dir" in action.params:
                    direction = action.params["dir"]
                    action_data["direction"] = direction.name
                    
                    # Calculate target position from direction
                    dx, dy = direction.delta
                    target_pos = (entity_pos[0] + dx, entity_pos[1] + dy)
                    action_data["target_pos"] = list(target_pos)
                    
                    # Calculate distance to all enemies after this move
                    if intel.visible_enemies:
                        min_dist_after = float('inf')
                        best_enemy_type = None
                        best_enemy_id = None
                        
                        for enemy in intel.visible_enemies:
                            e_pos = enemy.position
                            dist = abs(target_pos[0] - e_pos[0]) + abs(target_pos[1] - e_pos[1])
                            if dist < min_dist_after:
                                min_dist_after = dist
                                kind_str = enemy.kind.name if isinstance(enemy.kind, EntityKind) else str(enemy.kind).upper()
                                best_enemy_type = kind_str
                                best_enemy_id = enemy.id
                        
                        action_data["closest_enemy_after_move"] = {
                            "type": best_enemy_type,
                            "id": best_enemy_id,
                            "distance": min_dist_after
                        }
                    else:
                        action_data["closest_enemy_after_move"] = None

                # TOGGLE action: extract radar state
                elif action.type == ActionType.TOGGLE and "on" in action.params:
                    action_data["radar_on"] = action.params["on"]
                
                allowed_formatted.append(action_data)
            
            # Build unit block
            kind_str = entity.kind.name if isinstance(entity.kind, EntityKind) else str(entity.kind).upper()
            unit_block = {
                "entity_id": entity.id,
                "type": kind_str,
                "position": list(entity_pos),
                "missiles": getattr(entity, 'missiles', 0),
                "missile_range": missile_range,
                "enemies_in_range": enemies_in_range,
                "allowed_actions": allowed_formatted
            }
            units_data.append(unit_block)
        
        # Build batched prompt
        user_prompt = f"""Strategic Goal: {strategy.get('strategic_goal')}

TACTICAL SITUATION:
{json.dumps({"units": units_data}, indent=2)}

INSTRUCTIONS:
- For EACH unit, choose ONE action index from its allowed_actions list
- Shooting priority: AWACS > AIRCRAFT > SAM > DECOY
- MOVEMENT PRIORITY: Choose MOVE actions with SMALLEST "distance" in "closest_enemy_after_move"
- Prefer SHOOT if enemies in range and missiles available
- Move towards enemies and prefer shoot over move if enemies are in range
- If no enemies in range, move towards center of grid

Respond ONLY in JSON format:
{{"actions": [{{"entity_id": 1, "action_index": 0, "reasoning": "why move & shoot"}}, ...]}}
"""
        
        system_prompt = """You are a tactical air combat controller managing multiple units.
Fire missiles at enemy in range whenever possible.
Choose optimal actions for ALL units in one response.
Return ONLY valid JSON (no markdown)."""
        
        try:
            response = self.llm.complete(system_prompt, user_prompt, temperature=0.7)
            actions = self._parse_batched_actions(response, allowed_actions)
            logger.info(f"[Batch] Parsed {len(actions)} actions from LLM")
            
        except Exception as e:
            logger.error(f"[Batch] LLM call failed: {e}")
            actions = {}
        
        # Fill missing with random
        for entity_id, allowed in allowed_actions.items():
            if entity_id not in actions and allowed:
                actions[entity_id] = self.rng.choice(allowed)
                logger.warning(f"[Batch] Fallback random for entity {entity_id}")
        
        return actions

    def _parse_batched_actions(
        self,
        response: str,
        allowed_actions: Dict[int, List[Action]]
    ) -> Dict[int, Action]:
        """Parse batched action response."""
        json_str = self._extract_json(response)
        data = json.loads(json_str)
        
        actions = {}
        action_reasonings = {}
        for item in data.get("actions", []):
            entity_id = item.get("entity_id")
            action_idx = item.get("action_index")
            reasoning = item.get("reasoning")

            if entity_id not in allowed_actions:
                logger.warning(f"[Batch] Unknown entity_id {entity_id}")
                continue
            
            allowed = allowed_actions[entity_id]
            if 0 <= action_idx < len(allowed):
                actions[entity_id] = allowed[action_idx]
                action_reasonings[entity_id] = reasoning
                logger.debug(f"[Batch] Entity {entity_id} -> {allowed[action_idx].type.name} | Reasoning: {reasoning}")
            else:
                logger.warning(f"[Batch] Invalid index {action_idx} for entity {entity_id}")
        self.last_action_reasonings = action_reasonings
        return actions
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        
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

    def _init_llm_client(self):
        """Initialize LLM client based on provider."""
        if self.provider == "groq":
            if not self.api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            self.llm = GroqClient(api_key=self.api_key, model=self.model)
        elif self.provider == "ollama":
            self.llm = OllamaClient(model=self.model)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        logger.info(f"[{self.name}] Initialized {self.provider} client with model {self.model}")