import re
import json
from typing import List, Tuple, Optional

from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class StudentLLMAgentV1(Agent):
    """LLM-only decision agent. Always calls the LLM so the UI shows LLM Log."""

    RETRIES = 2  # retry once if JSON is bad

    # ---------- lifecycle ----------
    def _setup(self):
        """Initialize the OpenAI-compatible LLM client."""
        self.llm = OpenAIGomokuClient(model="qwen/qwen3-8b")

    # ---------- prompts ----------
    def _system_prompt(self, me: str) -> str:
        return (
            "You are a master-level Gomoku AI for an 8x8 board (0-indexed).\n"
            "Output ONLY one JSON object: {\"row\": <int>, \"col\": <int>}.\n"
            "The move MUST be in LEGAL_MOVES. No extra text.\n\n"
            "PRIORITY (choose the highest that applies):\n"
            "1) WIN NOW (complete five-in-a-row for YOU).\n"
            "2) BLOCK LOSS (opponent has immediate win next move; block unless you can win now).\n"
            "3) CREATE OPEN FOUR (forcing).\n"
            "4) CREATE DOUBLE OPEN THREE (fork).\n"
            "5) Extend strongest chains; 6) Center/Flexibility; Tie-breaker: earliest in LEGAL_MOVES.\n"
            "Before output, re-check (row,col) ∈ LEGAL_MOVES; otherwise pick the earliest move satisfying your highest rule.\n"
        )

    def _user_prompt(self, game_state, legal_moves: List[Tuple[int, int]]) -> str:
        me = self.player.value
        opp = 'O' if me == 'X' else 'X'
        board_str = game_state.format_board("standard")
        n = game_state.board_size
        legal_list = [[r, c] for (r, c) in legal_moves]
        return (
            f"BOARD {n}x{n} (0-indexed):\n{board_str}\n\n"
            f"You play as: {me}\nOpponent: {opp}\n"
            f"LEGAL_MOVES: {legal_list}\n"
            "Respond with JSON only."
        )

    # ---------- helpers ----------
    def _safe_extract_json(self, text) -> Optional[dict]:
        """Extract the first JSON object from model output; repair common issues."""
        if isinstance(text, dict):
            return text
        s = (text or "").strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r"\{.*?\}", s, re.DOTALL)  # non-greedy
        if not m:
            return None
        block = re.sub(r",\s*}", "}", m.group(0))  # fix trailing comma
        try:
            return json.loads(block)
        except Exception:
            return None

    def _fallback_move(self, game_state) -> Tuple[int, int]:
        legal = game_state.get_legal_moves()
        return legal[0] if legal else (game_state.board_size // 2, game_state.board_size // 2)

    # ---------- main ----------
    async def get_move(self, game_state):
        legal = game_state.get_legal_moves()
        if not legal:
            return self._fallback_move(game_state)

        messages = [
            {"role": "system", "content": self._system_prompt(self.player.value)},
            {"role": "user", "content": self._user_prompt(game_state, legal)},
        ]

        for _ in range(self.RETRIES):
            try:
                content = await self.llm.complete(
                    messages,
                    temperature=0.0,  # deterministic
                    top_p=0.9,
                    max_tokens=128,
                )
                data = self._safe_extract_json(content) or {}
                r, c = data.get("row"), data.get("col")
                try:
                    r = int(float(r))
                    c = int(float(c))
                except Exception:
                    r = c = None

                if (
                    isinstance(r, int) and isinstance(c, int)
                    and 0 <= r < game_state.board_size
                    and 0 <= c < game_state.board_size
                    and game_state.is_valid_move(r, c)
                ):
                    return (r, c)
            except Exception:
                # swallow and retry once; UI still shows LLM Log since we called the client
                pass

        return self._fallback_move(game_state)
