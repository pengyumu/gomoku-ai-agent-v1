import re
import json
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class StudentLLMAgentV1(Agent):
    def _setup(self):
        self.llm = OpenAIGomokuClient(model="qwen/qwen3-8b")

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

    def _has_open_three(self, board, p):
       
        size = len(board)
        EMPTY = Player.EMPTY.value
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for r in range(size):
            for c in range(size):
                for dr, dc in dirs:
                    cells = []
                    for k in range(5):
                        rr, cc = r + k*dr, c + k*dc
                        if 0 <= rr < size and 0 <= cc < size:
                            cells.append(board[rr][cc])
                        else:
                            cells = None
                            break
                    if not cells:
                        continue
                    if (cells[0] == EMPTY and cells[1] == p and cells[2] == p and
                        cells[3] == p and cells[4] == EMPTY):
                        return True
        return False

   
    def _find_immediate_win(self, board, legal_moves, p):
        size = len(board)
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for r, c in legal_moves:
            r, c = int(r), int(c)
            if board[r][c] != Player.EMPTY.value:
                continue
            board[r][c] = p 
            won = False
            for dr, dc in dirs:
                cnt = 1
                
                for sgn in (+1, -1):
                    rr, cc = r + sgn*dr, c + sgn*dc
                    while 0 <= rr < size and 0 <= cc < size and board[rr][cc] == p:
                        cnt += 1
                        rr += sgn*dr
                        cc += sgn*dc
                if cnt >= 5:
                    won = True
                    break
            board[r][c] = Player.EMPTY.value  
            if won:
                return (r, c)
        return None

    def analyze_board(self, game_state):
        board = game_state.board
        size = game_state.board_size
        player = self.player.value
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value
        EMPTY = Player.EMPTY.value

        
        stones = sum(1 for r in range(size) for c in range(size) if board[r][c] != EMPTY)

        return {
            "my_chain": self._get_max_chain_head(board, player),
            "rival_chain": self._get_max_chain_head(board, rival),
            "player": player,
            "rival": rival,
            "stones": stones,
        }

  
    def _sorted_moves_center_first(self, game_state):
        size = game_state.board_size
        cx = cy = (size - 1) / 2.0  # 8x8 时中心在(3.5,3.5)
        moves = list(game_state.get_legal_moves())
        moves.sort(key=lambda rc: (
            abs(rc[0] - cx) + abs(rc[1] - cy),           
            (rc[0] - cx)**2 + (rc[1] - cy)**2,         
            rc[0], rc[1]                                 
        ))
        return moves

    def _nearest_center_move(self, game_state):
        return self._sorted_moves_center_first(game_state)[0]


    async def get_move(self, game_state):
        stats = self.analyze_board(game_state)
        player = stats["player"]
        rival  = stats["rival"]

   
        legal_moves_center_first = self._sorted_moves_center_first(game_state)

      
        my_win_hint  = self._find_immediate_win(game_state.board, legal_moves_center_first, player)
        opp_win_hint = self._find_immediate_win(game_state.board, legal_moves_center_first, rival)

      
        base_policy = (
            "You are a master-level Gomoku AI on an 8×8 board (0-indexed: 0..7). "
            'Return ONLY one JSON exactly as {"row": <int>, "col": <int>} — on a SINGLE LINE, no extra text. '
            "DECISION RULES (in order): "
            "1) If REQUIRED_MOVES is NON-EMPTY, choose EXACTLY ONE move from REQUIRED_MOVES. "
            "2) Otherwise, follow the situation rule below."
        )

       
        if stats["my_chain"] >= 4:
            situation_rules = (
                "You have 4 consecutive stones. Place your stone to complete 5 in a line and win immediately."
            )
   
        elif stats["rival_chain"] >= 4 or self._has_open_three(game_state.board, rival):
            if opp_win_hint is not None:
                situation_rules = (
                    "CRITICAL THREAT: Opponent will win next at {opp_win_hint}. "
                    "Your next move must be {opp_win_hint}. "
                    "Select EXACTLY ONE coordinate from REQUIRED_MOVES and place your stone there. "
                    "Do NOT consider any other moves"
                )
            else:
                situation_rules = (
                    "Opponent shows a strong threat (four-in-line or open three). "
                    "Prioritize blocking; if two blocks exist, prefer the one that also extends your line."
                )
           
        elif stats["my_chain"] >= 3:
            situation_rules = (
                "You have 3 consecutive stones. You must extend to 4, preferably open-ended."
            )
        elif stats["my_chain"] >= 2 and stats["rival_chain"] <= 2:
            situation_rules = (
                "You have 2 consecutive stones. You must extend to 3, preferably open-ended."
            )
        elif stats["rival_chain"] >= 3:
            situation_rules = (
                "Opponent has 3 consecutive stones. Block them unless you can win immediately."
            )
        else:
            situation_rules = (
                "Play strategically: extend your longest chain with open ends, prefer central positions."
            )

      
        required_moves = []
        hint_lines = []

        if my_win_hint is not None:
            hint_lines.append(f"Analysis hint: your next step must be {my_win_hint}.")
            required_moves.append((int(my_win_hint[0]), int(my_win_hint[1])))

        if opp_win_hint is not None and (my_win_hint is None or tuple(opp_win_hint) != tuple(my_win_hint)):
            hint_lines.append(
                f"Analysis hint: opponent will win next at {opp_win_hint}. "
                f"You MUST block by choosing a move from REQUIRED_MOVES."
            )
            if my_win_hint is None:
                required_moves.append((int(opp_win_hint[0]), int(opp_win_hint[1])))

        
        required_moves = [(int(r), int(c)) for (r, c) in required_moves]
        hints_text = ("\n".join(hint_lines) + ("\n" if hint_lines else ""))

        system_prompt = base_policy + "\n" + situation_rules + ("\n" + hints_text if hints_text else "")

     
        board_str  = game_state.format_board("standard")
        board_size = game_state.board_size

        user_content = (
            f"BOARD {board_size}x{board_size}:\n{board_str}\n\n"
            f"You play as: {player}\nOpponent: {rival}\n"
            f"LEGAL_MOVES (row,col): {legal_moves_center_first}\n"
            f"REQUIRED_MOVES (row,col): {required_moves}\n"
            'Output ONE JSON: {"row": <int>, "col": <int>}.'
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        content = await self.llm.complete(messages)

      
        move = None
        if isinstance(content, str):
            m = re.search(r'\{\s*"row"\s*:\s*-?\d+\s*,\s*"col"\s*:\s*-?\d+\s*\}', content)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    move = (int(obj["row"]), int(obj["col"]))
                except Exception:
                    move = None

        if move is not None and game_state.is_valid_move(*move):
            return move

       
        return self._nearest_center_move(game_state)
