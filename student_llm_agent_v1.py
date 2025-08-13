import re
import json
import asyncio
import random
from typing import Tuple, List, Dict

from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player
from gomoku.arena import GomokuArena
from gomoku.utils import ColorBoardFormatter
from gomoku.llm.openai_client import OpenAIGomokuClient
from gomoku.core.game_logic import GomokuGame


class StudentLLMAgentV1(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    def _setup(self):
        self.system_prompt = self._create_system_prompt()
        self.llm_client = OpenAIGomokuClient(model="Qwen/Qwen3-8B")
    def _create_system_prompt(self) -> str:
        return (
        "You are a master-level Gomoku AI with perfect tactical vision on an 8×8 board (0-indexed).\n"
        "Your goal is to win by forming five consecutive stones in any direction. Never play on occupied cells.\n\n"
        "### CRITICAL INSTRUCTIONS ###\n"
        "1. You MUST output ONLY one valid JSON object: {\"row\": <int>, \"col\": <int>}\n"
        "2. No explanation, no markdown, no extra text. Only JSON.\n"
        "3. The move MUST be in LEGAL_MOVES.\n\n"
        "### CORE STRATEGY ###\n"
        "🔥 BLOCK OPPONENT'S OPEN THREE IMMEDIATELY\n"
        "   - An open three (e.g., '.XXX.' or 'XX.X') can become open four next move.\n"
        "   - If you don't block it now, you will be forced to block it later — and lose your turn.\n"
        "   - The ONLY exception: if you can WIN THIS TURN or CREATE YOUR OWN OPEN FOUR.\n\n"
        "### MOVE HIERARCHY ###\n"
        "1. WIN NOW: Can you complete 5 in a row? → play it.\n"
        "2. BLOCK OPPONENT'S OPEN FOUR: → must block.\n"
        "3. BLOCK OPPONENT'S OPEN THREE: → must block, unless you have a stronger threat.\n"
        "   This is PREVENTIVE DEFENSE — stop the threat before it happens.\n\n"
        "4. ESCALATE YOUR OWN THREAT: Can you upgrade your open three to open four? → do it if opponent has no open three .\n"
        "5. CREATE A FORK: Two open threes → better to have.\n"
        "6. EXTEND CHAINS: Prefer open threes.\n"
        "7. CONTROL CENTER: Prefer (3,3)-(4,4).\n"
        "8. FINAL TIEBREAKER: Pick the earliest legal move.\n\n"
        "### PATTERN GUIDE ###\n"
        "- Open Three: '.XXX.', 'XX.X', 'X.XX' → DANGEROUS — will become open four\n"
        "- Open Four: '.XXXX', 'XXX.X' → MUST RESPOND — game over if ignored\n"
        "- Broken Three: 'XX.XX' → not immediate threat\n\n"
        "### THINKING PROTOCOL ###\n"
        "Before choosing, ask:\n"
        "1. Can I win this turn?\n"
        "2. Does opponent have open four? → block.\n"
        "3. Does opponent have open three? → block (unless I can win or make open four).\n"
        "4. Can I upgrade my own open three to open four? → do it.\n\n"
        "⚠️ REMEMBER: If you let opponent reach open four, you lose control.\n"
        "Block open threes EARLY to maintain initiative.\n\n"
        "### OUTPUT FORMAT ###\n"
        "Write analysis in <analysis>...</analysis>, then output JSON on a new line.\n"
        "Example:\n"
        "<analysis>\n"
        "Opponent has open three at (2,3)-(4,3): . O O O .\n"
        "If they play (5,3), it becomes open four → I must block next turn.\n"
        "I cannot win this turn, so I MUST block now at (5,3) to prevent loss of initiative.\n"
        "</analysis>\n"
        "{\"row\": 5, \"col\": 3}"
    )
    
    '''def _create_system_prompt(self) -> str:
        return (
            "You are a master-level Gomoku AI with perfect tactical vision on an 8×8 board (0-indexed).\n"
            "Your goal is to win by forming five consecutive stones in any direction. Never play on occupied cells.\n\n"
            "### CRITICAL INSTRUCTIONS ###\n"
            "1. You MUST output ONLY one valid JSON object: {\"row\": <int>, \"col\": <int>}\n"
            "2. No explanation, no markdown, no extra text. Only JSON.\n"
            "3. The move MUST be in LEGAL_MOVES.\n\n"
            "### MOVE HIERARCHY (APPLY IN ORDER) ###\n"
            "Evaluate these conditions strictly from top to bottom:\n"
            "1️⃣ WIN NOW: If you can complete a five-in-a-row this turn → play that move.\n"
            "2️⃣ BLOCK IMMINENT LOSS: If opponent has any four-in-a-row (open or closed) → block it immediately.\n"
            "3️⃣ CREATE AN OPEN FOUR: Place a stone to form a sequence of four of your stones with empty ends.\n"
            "4️⃣ SET UP A DOUBLE THREAT: Create two simultaneous open threes (fork), forcing a loss for opponent.\n"
                "5️⃣ EXTEND STRONG CHAINS: Prioritize extending your longest existing lines (especially open threes or broken fours).\n"
                "6️⃣ CONTROL CENTER & FLEXIBILITY: Prefer central squares (e.g., near (3,3) to (4,4)) and positions that allow multiple future directions.\n"
                "7️⃣ FINAL TIEBREAKER: Choose the move that appears earliest in the LEGAL_MOVES list.\n\n"
                "### THINKING PROTOCOL ###\n"
                "Before choosing, mentally simulate:\n"
                "- For each candidate move in LEGAL_MOVES:\n"
                "  a) Would this win me the game?\n"
                "  b) Does this stop opponent from winning next turn?\n"
                "  c) Does this create a new open four or double threat?\n"
                "Prioritize based on the hierarchy above.\n\n"
                "⚠️ SELF-CHECK: If your chosen move is NOT in LEGAL_MOVES, DO NOT output it.\n"
            "Instead, scan LEGAL_MOVES in order and pick the first one that best satisfies the hierarchy above.\n"
            "This ensures robustness even if internal thinking fails.\n"
        )'''

    def _build_user_prompt(self, game_state: GameState, legal_moves):
        board_str = game_state.format_board("standard")
        you = game_state.current_player.value
        opp = 'O' if you == 'X' else 'X'
        legal_list = [[r, c] for (r, c) in legal_moves]
        return (
            f"### CURRENT GAME STATE ###\n"
            f"BOARD (8x8, 0-indexed):\n{board_str}\n\n"
            f"you_play: {you}\n"
            f"opponent: {opp}\n\n"
            f"Available moves (row, col): {legal_list}\n\n"
            f"Choose your move according to the strategy rules. Output JSON only."
        )
    
    def _safe_extract_json(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        block = m.group(0)
        try:
            return json.loads(block)
        except Exception:
            cleaned = re.sub(r",\s*}", "}", block)
            try:
                return json.loads(cleaned)
            except Exception:
                return None

    def _fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        legal = game_state.get_legal_moves()
        if not legal:
            center = game_state.board_size // 2
            return (center, center)
        center = game_state.board_size // 2
        return min(legal, key=lambda rc: abs(rc[0]-center) + abs(rc[1]-center))

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            return self._fallback_move(game_state)

        if not hasattr(self, "llm_client") or self.llm_client is None:
            return self._fallback_move(game_state)

        try:
            user_prompt = self._build_user_prompt(game_state, legal_moves)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
            response = await self.llm_client.complete(
                messages,
                temperature=0.0,
                top_p=0.9,
                max_tokens=128,
            )
            data = self._safe_extract_json(response) or {}
            r, c = data.get("row"), data.get("col")
            if isinstance(r, int) and isinstance(c, int) and (r, c) in set(legal_moves):
                return (r, c)
            return self._fallback_move(game_state)
        except Exception:
            return self._fallback_move(game_state)
