import re
import json
from typing import List, Tuple, Optional
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player

class StudentLLMAgentV1(Agent):
 
    RETRIES = 2
    DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

    def _setup(self):
        """
        Initialize the agent by setting up the language model client.
        This method is called once when the agent is created.
        """
        # Create an OpenAI-compatible client using the Gemma2 model for move generation
        self.llm = OpenAIGomokuClient(model="qwen/qwen3-8b")

    def _log(self, *a):
        if self.DEBUG:
            print("[StudentLLMAgentV1]", *a)

    def _safe_extract_json(self, text) -> Optional[dict]:
        """
        Extract first JSON object from model output; repair common issues.
        Accepts dict or string; returns dict or None.
        """
        if isinstance(text, dict):
            return text

        s = (text or "").strip()
        # 1) try direct parse
        try:
            return json.loads(s)
        except Exception:
            pass

        # 2) find the first JSON object (non-greedy)
        m = re.search(r"\{.*?\}", s, re.DOTALL)
        if not m:
            return None

        block = m.group(0)
        # repair trailing comma
        block = re.sub(r",\s*}", "}", block)

        try:
            return json.loads(block)
        except Exception:
            return None
        
# -------------------- board helpers --------------------

    def _try_cell(self, gs, r: int, c: int) -> Optional[str]:
        """
        Return '.', 'X', or 'O' if discoverable; else None.
        Tries direct internal board first, then falls back to formatted board parsing.
        """
        # Try direct grid-like attributes
        for name in ("board", "grid", "cells", "state", "matrix"):
            b = getattr(gs, name, None)
            if b is not None:
                try:
                    v = b[r][c]
                    if v in ('.', 'X', 'O'):
                        return v
                    if v in (0, 1, 2):  # numeric encodings common in some engines
                        return {0: '.', 1: 'X', 2: 'O'}[v]
                except Exception:
                    pass

        # Fallback: parse from formatted string
        try:
            s = gs.format_board("standard")
            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            n = gs.board_size
            rows = lines[-n:]
            row = rows[r]
            symbols = [ch for ch in row if ch in ('.', 'X', 'O')]
            if len(symbols) >= n:
                return symbols[c]
        except Exception:
            pass

        return None

    def _count_dir(self, gs, r: int, c: int, dr: int, dc: int, me: str) -> int:
        n = gs.board_size
        rr, cc = r + dr, c + dc
        k = 0
        while 0 <= rr < n and 0 <= cc < n and self._try_cell(gs, rr, cc) == me:
            k += 1
            rr += dr
            cc += dc
        return k

    def _is_five_if_play(self, gs, r: int, c: int, me: str) -> bool:
        """If placing me at (r,c), do we form five-in-a-row?"""
        if self._try_cell(gs, r, c) != '.':
            return False
        for dr, dc in self.DIRECTIONS:
            a = self._count_dir(gs, r, c, -dr, -dc, me)
            b = self._count_dir(gs, r, c,  dr,  dc, me)
            if a + 1 + b >= 5:
                return True
        return False

    def _open_four_if_play(self, gs, r: int, c: int, me: str) -> bool:
        """
        After placing at (r,c), do we have exactly 4-in-line with at least one open end?
        This is a forcing shape: opponent must respond.
        """
        if self._try_cell(gs, r, c) != '.':
            return False
        n = gs.board_size
        for dr, dc in self.DIRECTIONS:
            a = self._count_dir(gs, r, c, -dr, -dc, me)
            b = self._count_dir(gs, r, c,  dr,  dc, me)
            total = a + 1 + b
            if total == 4:
                end1 = (r - (a + 1) * dr, c - (a + 1) * dc)
                end2 = (r + (b + 1) * dr, c + (b + 1) * dc)

                def open_end(rr, cc):
                    return 0 <= rr < n and 0 <= cc < n and self._try_cell(gs, rr, cc) == '.'

                if open_end(*end1) or open_end(*end2):
                    return True
        return False

    def _double_open_three_if_play(self, gs, r: int, c: int, me: str) -> bool:
        """
        Does (r,c) create two distinct open-threes (a fork)?
        """
        if self._try_cell(gs, r, c) != '.':
            return False
        n = gs.board_size
        dirs = 0
        for dr, dc in self.DIRECTIONS:
            a = self._count_dir(gs, r, c, -dr, -dc, me)
            b = self._count_dir(gs, r, c,  dr,  dc, me)
            total = a + 1 + b
            if total == 3:
                end1 = (r - (a + 1) * dr, c - (a + 1) * dc)
                end2 = (r + (b + 1) * dr, c + (b + 1) * dc)

                def open_end(rr, cc):
                    return 0 <= rr < n and 0 <= cc < n and self._try_cell(gs, rr, cc) == '.'

                if open_end(*end1) and open_end(*end2):
                    dirs += 1
        return dirs >= 2

    # -------------------- tactical pre-move --------------------

    def _tactical_move(self, gs, legal_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Hard rules in priority:
        1) Win now
        2) Block opponent win now
        3) Create open four
        4) Create double open three
        """
        me = self.player.value              # 'X' or 'O'
        opp = 'O' if me == 'X' else 'X'

        # If board unreadable, skip gracefully
        if self._try_cell(gs, 0, 0) is None:
            self._log("Board unreadable → skip tactical premove")
            return None

        # 1) win now
        for r, c in legal_moves:
            if self._is_five_if_play(gs, r, c, me):
                self._log("Tactical: WIN NOW", (r, c))
                return (r, c)

        # 2) block opponent's immediate win
        for r, c in legal_moves:
            if self._is_five_if_play(gs, r, c, opp):
                self._log("Tactical: BLOCK LOSS", (r, c))
                return (r, c)

        # 3) create open four
        for r, c in legal_moves:
            if self._open_four_if_play(gs, r, c, me):
                self._log("Tactical: OPEN FOUR", (r, c))
                return (r, c)

        # 4) create double open three
        for r, c in legal_moves:
            if self._double_open_three_if_play(gs, r, c, me):
                self._log("Tactical: DOUBLE THREE", (r, c))
                return (r, c)

        return None

    # -------------------- prompts --------------------

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

    def _user_prompt(self, gs, legal_moves: List[Tuple[int, int]]) -> str:
        me = self.player.value
        opp = 'O' if me == 'X' else 'X'
        board = gs.format_board("standard")
        n = gs.board_size
        legal_list = [[r, c] for (r, c) in legal_moves]
        return (
            f"BOARD {n}x{n} (0-indexed):\n{board}\n\n"
            f"You play as: {me}\nOpponent: {opp}\n"
            f"LEGAL_MOVES: {legal_list}\n"
            "Return best move as JSON only."
        )

    # -------------------- main --------------------

    async def get_move(self, game_state):
        legal = game_state.get_legal_moves()
        if not legal:
            n = game_state.board_size
            return (n // 2, n // 2)

        # 1) Tactical first (hard rules)
        tmove = self._tactical_move(game_state, legal)
        if tmove is not None:
            return tmove

        # 2) LLM decision with small retry
        sys = self._system_prompt(self.player.value)
        usr = self._user_prompt(game_state, legal)
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

        for attempt in range(self.RETRIES):
            try:
                self._log(f"Calling LLM (attempt {attempt + 1})…")
                content = await self.llm.complete(
                    messages,
                    temperature=0.0,
                    top_p=0.9,
                    max_tokens=128,
                )
                self._log("LLM raw:", repr(content)[:160])

                data = self._safe_extract_json(content) or {}
                r, c = data.get("row"), data.get("col")

                # coerce "3"/3.0 -> 3
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
                    self._log("LLM move →", (r, c))
                    return (r, c)

                self._log("Invalid LLM move:", data)

            except Exception as e:
                self._log("LLM exception:", e)

        # 3) Neutral fallback: first legal (avoid center bias)
        return legal[0]
