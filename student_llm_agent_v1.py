# Import necessary modules for regular expressions, JSON parsing, and the Gomoku framework
import re
import json
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class StudentLLMAgentV1(Agent):
    """
    A Gomoku AI agent that uses a language model to make strategic moves.
    Inherits from the base Agent class provided by the Gomoku framework.
    """

    def _setup(self):
        """
        Initialize the agent by setting up the language model client.
        This method is called once when the agent is created.
        """
        # Create an OpenAI-compatible client using the Gemma2 model for move generation
        self.llm = OpenAIGomokuClient(model="qwen/qwen3-8b")

    async def get_move(self, game_state):
        """
        Generate the next move for the current game state using an LLM.

        Args:
            game_state: Current state of the Gomoku game board

        Returns:
            tuple: (row, col) coordinates of the chosen move
        """
        # Get the current player's symbol (e.g., 'X' or 'O')
        player = self.player.value

        # Determine the opponent's symbol by checking which player we are
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        # Convert the game board to a human-readable string format
        board_str = game_state.format_board("standard")
        board_size = game_state.board_size

        # Prepare the conversation messages for the language model
        messages = [
    {
        "role": "system",
        "content":
            "You are a master-level Gomoku AI on an 8×8 board (0-indexed: rows/cols 0..7).\n"
            "Return ONLY one JSON object exactly as {\"row\": <int>, \"col\": <int>} — no extra text.\n"
            "Your move MUST be in LEGAL_MOVES and on an empty cell.\n"
            "Symbols: empty '.', black 'X'. White is 'O'; NOTE: some boards use the digit '0' for white — treat '0' exactly as 'O'.\n\n"
            "POLICY — evaluate in this order and choose the first applicable rule:\n"
            "1) WIN NOW — if any move makes five in a row for YOU, play it.\n"
            "2) BLOCK LOSS — if opponent can win next move, block that line (unless you can win now).\n"
            "3) FORCING FOUR — create a four that forces a reply ('.XXXX', 'XXXX.', 'XXX.X').\n"
            "4) OPEN THREE FIRST — if no immediate threats, PREFER making an OPEN THREE for yourself ('.OOO.' or 'OO.O' / '.XXX.' or 'XX.X' depending on your color) near your strongest chain.\n"
            "5) DENY OPPONENT OPEN THREE — if you cannot win or force, block their open three.\n"
            "6) DOUBLE THREAT — two independent threats (e.g., two open threes) to force a win.\n"
            "7) SHAPE & CENTER — otherwise extend your longest line with open ends; prefer central squares around (3,3)–(4,4).\n"
            "8) TIE-BREAKER — if still tied, pick the earliest move in LEGAL_MOVES.\n"
            "SELF-CHECK: ensure (row,col) ∈ LEGAL_MOVES; if not, scan LEGAL_MOVES in order and output the first move that satisfies the highest rule."
    },
    {
        "role": "user",
        "content":
            f"BOARD {board_size}x{board_size} (0-indexed):\n{board_str}\n\n"
            f"You play as: {player} (remember: if you are white, '0' on the board means your stones).\n"
            f"Opponent: {rival}\n"
            f"LEGAL_MOVES (row,col): {game_state.get_legal_moves()}\n\n"
            "Apply the policy and reply with JSON only."
    },
]


        # Send the messages to the language model and get the response
        content = await self.llm.complete(messages)

        # Parse the LLM response to extract move coordinates
        try:
            # Use regex to find JSON-like content in the response
            if m := re.search(r"{[^}]+}", content, re.DOTALL):
                # Parse the JSON to extract row and column
                move = json.loads(m.group(0))
                row, col = (move["row"], move["col"])

                # Validate that the proposed move is legal
                if game_state.is_valid_move(row, col):
                    return (row, col)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, continue to fallback strategy
            pass

        # Fallback: if LLM response is invalid, choose the first available legal move
        return game_state.get_legal_moves()[0]
