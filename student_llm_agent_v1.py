import re
import json
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class StudentLLMAgentV2(Agent):
    def _setup(self):
        self.llm = OpenAIGomokuClient(model="google/gemma-2-9b-it")

    # ---------- 连子统计：仅从“链头”起数，覆盖横/竖/两斜 ----------
    def _get_max_chain_head(self, board, p):
        size = len(board)
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        best = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] != p:
                    continue
                for dr, dc in dirs:
                    pr, pc = r - dr, c - dc
                    # 仅从链头开始（前一格不是同色或越界）
                    if 0 <= pr < size and 0 <= pc < size and board[pr][pc] == p:
                        continue
                    rr, cc, length = r, c, 0
                    while 0 <= rr < size and 0 <= cc < size and board[rr][cc] == p:
                        length += 1
                        rr += dr
                        cc += dc
                    if length > best:
                        best = length
        return best

    # ---------- 单手必胜检测：尝试每个合法点，看是否能形成5连 ----------
    def _has_immediate_win(self, board, legal_moves, p):
        size = len(board)
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for r, c in legal_moves:
            board[r][c] = p  # 临时落子
            won = False
            for dr, dc in dirs:
                cnt = 1
                # 双向累计
                for sgn in (+1, -1):
                    rr, cc = r + sgn*dr, c + sgn*dc
                    while 0 <= rr < size and 0 <= cc < size and board[rr][cc] == p:
                        cnt += 1
                        rr += sgn*dr
                        cc += sgn*dc
                if cnt >= 5:
                    won = True
                    break
            board[r][c] = Player.EMPTY.value  # 撤回
            if won:
                return (r, c)
        return None

    def analyze_board(self, game_state):
        board = game_state.board
        size = game_state.board_size
        player = self.player.value
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value
        EMPTY = Player.EMPTY.value

        # 统计当前已落子数量（用于“开局走中路”的启发式）
        stones = sum(1 for r in range(size) for c in range(size) if board[r][c] != EMPTY)

        return {
            "my_chain": self._get_max_chain_head(board, player),
            "rival_chain": self._get_max_chain_head(board, rival),
            "player": player,
            "rival": rival,
            "stones": stones,
        }

    # —— 按“离中心距离”对合法点排序 / 取最近点 ——
    def _sorted_moves_center_first(self, game_state):
        size = game_state.board_size
        cx = cy = (size - 1) / 2.0  # 8x8 时中心在(3.5,3.5)
        moves = list(game_state.get_legal_moves())
        moves.sort(key=lambda rc: (
            abs(rc[0] - cx) + abs(rc[1] - cy),            # 先按曼哈顿距离
            (rc[0] - cx)**2 + (rc[1] - cy)**2,           # 再按欧氏距离平方做细分
            rc[0], rc[1]                                  # 最后稳定排序
        ))
        return moves

    def _nearest_center_move(self, game_state):
        return self._sorted_moves_center_first(game_state)[0]

    async def get_move(self, game_state):
        stats = self.analyze_board(game_state)
        player = stats["player"]
        rival = stats["rival"]

        # ---------- 1) 规则前置：自己一手赢 / 立即堵 ----------
        legal_moves = list(game_state.get_legal_moves())

        # 自己一手致胜
        win_move = self._has_immediate_win(game_state.board, legal_moves, player)
        if win_move is not None:
            return win_move

        # 对手一手致胜 -> 立刻堵（与对手的必胜点同一空位）
        block_move = self._has_immediate_win(game_state.board, legal_moves, rival)
        if block_move is not None and game_state.is_valid_move(*block_move):
            return block_move

        # ---------- 2) 调用 LLM 的策略提示 ----------
        base_policy = (
            "You are a master-level Gomoku AI on an 8×8 board (0-indexed: rows/cols 0..7). "
            "Return ONLY one JSON object exactly as {\"row\": <int>, \"col\": <int>} — no extra text. "
            "Numbers must be integers. The move MUST be one of LEGAL_MOVES and on an empty cell.\n"
            "If there is no immediate win or forced block, prefer the legal move closest to the board center.\n"
        )

        if stats["my_chain"] >= 4:
            situation_rules = (
                "You have 4 consecutive stones (horizontal / vertical / diagonal). "
                "Place your stone to complete 5 in a line and win immediately.\n"
            )
        elif stats["rival_chain"] >= 4:
            situation_rules = (
                "Opponent has 4 consecutive stones (horizontal / vertical / diagonal). "
                "Immediately block to prevent them from winning.\n"
            )
        elif stats["my_chain"] < 2 and stats["rival_chain"] < 2:
            situation_rules = "Early game: extend towards the center; avoid isolated edge moves.\n"
        elif stats["my_chain"] >= 3:
            situation_rules = (
                "You have 3 consecutive stones (horizontal / vertical / diagonal). "
                "Extend to 4, preferably open-ended.\n"
            )
        elif stats["rival_chain"] >= 3:
            situation_rules = (
                "Opponent has 3 consecutive stones (horizontal / vertical / diagonal). "
                "Block them unless you can win immediately.\n"
            )
        else:
            situation_rules = (
                "Play strategically: extend your longest chain with open ends, prefer central positions.\n"
            )

        system_prompt = base_policy + situation_rules

        board_str = game_state.format_board("standard")
        board_size = game_state.board_size

        # 把 LEGAL_MOVES 以“中心优先”的顺序提供给 LLM，进一步引导它选中路
        legal_moves_center_first = self._sorted_moves_center_first(game_state)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"BOARD {board_size}x{board_size}:\n{board_str}\n\n"
                    f"You play as: {player}\nOpponent: {rival}\n"
                    f"LEGAL_MOVES (row,col): {legal_moves_center_first}\n"
                )
            },
        ]

        content = await self.llm.complete(messages)

        # ---------- 3) 解析 LLM 的落子 ----------
        try:
            if m := re.search(r"{[^}]+}", content, re.DOTALL):
                move = json.loads(m.group(0))
                row, col = move["row"], move["col"]
                if game_state.is_valid_move(row, col):
                    return (row, col)
        except Exception:
            pass

        # ---------- 4) 兜底：离中心最近 ----------
        return self._nearest_center_move(game_state)
