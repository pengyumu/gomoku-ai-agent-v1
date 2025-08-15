import re
import json
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class StudentLLMAgentV3(Agent):
    def _setup(self):
        """Initialize the LLM client."""
        self.llm = OpenAIGomokuClient(model="google/gemma-2-9b-it")

    def analyze_board(self, game_state):
        """Analyze board and return key stats for strategy selection."""
        board = game_state.board
        size = game_state.board_size
        player = self.player.value
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value
        EMPTY = Player.EMPTY.value

        def get_max_chain(p):
            max_chain = 0
            directions = [(1,0), (0,1), (1,1), (1,-1)]
            for r in range(size):
                for c in range(size):
                    for dr, dc in directions:
                        length = 0
                        rr, cc = r, c
                        while 0 <= rr < size and 0 <= cc < size and board[rr][cc] == p:
                            length += 1
                            rr += dr
                            cc += dc
                        max_chain = max(max_chain, length)
            return max_chain

        my_chain = get_max_chain(player)
        rival_chain = get_max_chain(rival)
        return {
            "my_chain": my_chain,
            "rival_chain": rival_chain,
            "player": player,
            "rival": rival
        }

    async def get_move(self, game_state):
        """Generate next move purely via LLM, but with dynamic prompt."""
        stats = self.analyze_board(game_state)
        player = stats["player"]
        rival = stats["rival"]

        # Base policy text
        base_policy = (
            "You are a master-level Gomoku AI on an 8×8 board (0-indexed: rows/cols 0..7). "
            "Return ONLY one JSON object exactly as {\"row\": <int>, \"col\": <int>} — no extra text. "
            "Numbers must be integers. The move MUST be one of LEGAL_MOVES and on an empty cell.\n"
        )

        # Dynamic rule injection
        if stats["my_chain"] < 2 and stats["rival_chain"] < 2:
            # Early game expansion rule
            situation_rules = (
                "Both you and opponent have less than 2 stones in a row. "
                "Do NOT block opponent. Prioritize extending your own stones to build chains "
                "towards the center. Avoid random isolated moves.\n"
            )
        elif stats["my_chain"] >= 4:
            situation_rules = (
                "You have 4 in a row. Immediately win by making 5 in a row.\n"
            )
        elif stats["rival_chain"] >= 4:
            situation_rules = (
                "Opponent has 4 in a row. Immediately block to prevent them from winning.\n"
            )
        elif stats["my_chain"] >= 3:
            situation_rules = (
                "You have 3 in a row. Extend to 4, preferably open-ended.\n"
            )
        elif stats["rival_chain"] >= 3:
            situation_rules = (
                "Opponent has 3 in a row. Block them unless you can win immediately.\n"
            )
        else:
            situation_rules = (
                "Play strategically: extend your longest chain with open ends, "
                "prefer central positions, and avoid helping opponent.\n"
            )

        # Full system prompt
        system_prompt = base_policy + situation_rules

        # Format board
        board_str = game_state.format_board("standard")
        board_size = game_state.board_size

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"BOARD {board_size}x{board_size}:\n{board_str}\n\n"
                    f"You play as: {player}\nOpponent: {rival}\n"
                    "LEGAL_MOVES (row,col): {moves}\n"
                ).format(moves=game_state.get_legal_moves())
            },
        ]

        # Call LLM
        content = await self.llm.complete(messages)

        # Parse JSON move
        try:
            if m := re.search(r"{[^}]+}", content, re.DOTALL):
                move = json.loads(m.group(0))
                row, col = move["row"], move["col"]
                if game_state.is_valid_move(row, col):
                    return (row, col)
        except Exception:
            pass

        # Fallback: first legal move
        return game_state.get_legal_moves()[0]
