import re
import json
from typing import List, Tuple, Optional

from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class StudentLLMAgentV1(Agent):
    """
    LLM-only Gomoku agent.
    Always calls the LLM; no tactical pre-moves or search.
    """

    RETRIES = 2  # retry once if the JSON is malformed

    # --------------- lifecycle ----------------
    def _setup(self):
        """Initialize the language model client."""
        self.llm = OpenAIGomokuClient(model="qwen/qwen3-8b")

    # --------------- helpers ------------------
    def _extract_move(self, content: str) -> Optional[Tuple[int, int]]:
        """
        Extract (row, col) from model output.
        - Accepts either a full JSON string or any text containing {"row": ...,"col": ...}
        - Coerces numeric strings/floats to ints to avoid type-mismatch later
        """
        if not isinstance(content, str):
            # Some runtimes might pass a dict already
            try:
                r = int(float(content.get("row")))
                c = int(float(content.get("col")))
                return (r, c)
            except Exception:
                return None

        s = content.strip()

        # 1) Try direct JSON object first
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "row" in obj and "col" in obj:
                return (int(float(obj["row"])), int(float(obj["col"])))
        except Exception:
            pass

        # 2) Regex for a strict JSON object {"row": <num>, "col": <num>}
        m = re.search(
            r'\{\s*"row"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"col"\s*:\s*(-?\d+(?:\.\d+)?)\s*\}',
            s,
            flags=re.DOTALL,
        )
        if m:
            try:
                r = int(float(m.group(1)))
                c = int(float(m.group(2)))
                return (r, c)
            except Exception:
                return None

        # 3) Fallback: first {...} block, with minor trailing-comma repair
        m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
        if m:
            block = re.sub(r",\s*}", "}", m.group(0))
            try:
                obj = json.loads(block)
                if isinstance(obj, dict) and "row" in obj and "col" in obj:
                    return (int(float(obj["row"])), int(float(obj["col"])))
            except Exception:
                return None

        return None

    def _fallback_move(self, game_state) -> Tuple[int, int]:
        legal = game_state.get_legal_moves()
        return legal[0] if legal else (game_state.board_size // 2, game_state.board_size // 2)

    # --------------- main ---------------------
    async def get_move(self, game_state):
        legal_moves_raw = game_state.get_legal_moves()
        if not legal_moves_raw:
            return self._fallback_move(game_state)

        # 规范化 LEGAL_MOVES：全部强制转成整数元组，避免后续比较时出现 str vs int
        legal_moves: List[Tuple[int, int]] = []
        for r, c in legal_moves_raw:
            try:
                legal_moves.append((int(r), int(c)))
            except Exception:
                # 如果某个坐标不是数字，直接跳过，留给引擎的合法校验兜底
                continue

        # 当前玩家/对手、棋盘
        player = self.player.value  # 'X' or 'O'
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value
        board_str = game_state.format_board("standard")
        n = game_state.board_size

        # -------- Prompt (只改提示词，不引入策略代码) --------
        messages = [
            {
                "role": "system",
                "content":
                    "You are a master-level Gomoku AI on an 8×8 board (0-indexed: rows/cols 0..7).\n"
                    "Return ONLY one JSON object exactly as {\"row\": <int>, \"col\": <int>} — no extra text.\n"
                    "Your move MUST be in LEGAL_MOVES and on an empty cell.\n\n"
                    "EVALUATE IN THIS EXACT ORDER AND CHOOSE THE FIRST APPLICABLE RULE:\n"
                    "1) WIN NOW — if any move makes five in a row for YOU, play it.\n"
                    "2) BLOCK LOSS — if the opponent can win next move, block that line (unless you can win now).\n"
                    "3) FORCING FOUR — create a forcing four ('.XXXX', 'XXXX.', or 'XXX.X').\n"
                    "4) OPEN THREE FIRST — if there is no immediate win/loss threat, prefer making an OPEN THREE for yourself ('.OOO.' or 'OO.O' / '.XXX.' or 'XX.X' depending on your color) near your strongest chain.\n"
                    "5) DENY OPPONENT OPEN THREE — if you cannot win or force, block their open three.\n"
                    "6) DOUBLE THREAT — two independent threats (e.g., two open threes) to force a win.\n"
                    "7) SHAPE & CENTER — otherwise extend your longest line with open ends; prefer central squares around (3,3)–(4,4).\n"
                    "8) TIE-BREAKER — if still tied, pick the earliest move in LEGAL_MOVES.\n"
                    "SELF-CHECK: ensure (row,col) ∈ LEGAL_MOVES; if not, scan LEGAL_MOVES in order and output the first move that satisfies the highest rule."
            },
            {
                "role": "user",
                "content":
                    f"BOARD {n}x{n} (0-indexed):\n{board_str}\n\n"
                    f"You play as: {player}\nOpponent: {rival}\n"
                    f"LEGAL_MOVES (row,col): {legal_moves}\n\n"
                    "Apply the policy and reply with JSON only."
            },
        ]

        # 调用 LLM，并做稳健解析 + 类型强转
        for _ in range(self.RETRIES):
            try:
                content = await self.llm.complete(
                    messages,
                    temperature=0.0,   # deterministic
                    top_p=0.9,
                    max_tokens=128,
                )

                mv = self._extract_move(content)
                if mv is None:
                    continue

                row, col = mv  # 已强制转 int
                # 二次合法性检查（行列范围 + 是否在合法集合内）
                if (
                    isinstance(row, int) and isinstance(col, int)
                    and 0 <= row < n and 0 <= col < n
                    and (row, col) in set(legal_moves)
                    and game_state.is_valid_move(row, col)
                ):
                    return (row, col)

            except Exception:
                # 吞掉单次异常，尝试下一次；最终仍会 fallback
                pass

        # 兜底：中性回退到第一个合法步（不做中心偏置）
        return self._fallback_move(game_state)
