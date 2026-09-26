from __future__ import annotations

from dataclasses import dataclass

from .base import Environment

ROWS, COLS = 6, 7


@dataclass(frozen=True, slots=True)
class ConnectFourState:
    """Immutable row-major board. Row 0 is the top of the board."""

    board: tuple[int, ...] = (0,) * (ROWS * COLS)
    player: int = 1
    winner: int = 0
    moves: int = 0

    def at(self, row: int, col: int) -> int:
        return self.board[row * COLS + col]


class ConnectFour(Environment[ConnectFourState, int]):
    def initial_state(self) -> ConnectFourState:
        return ConnectFourState()

    def legal_actions(self, state: ConnectFourState) -> tuple[int, ...]:
        if self.is_terminal(state):
            return ()
        return tuple(col for col in range(COLS) if state.at(0, col) == 0)

    def step(self, state: ConnectFourState, action: int) -> ConnectFourState:
        if self.is_terminal(state):
            raise ValueError("cannot move in a terminal state")
        if action not in range(COLS) or state.at(0, action) != 0:
            raise ValueError(f"illegal column: {action}")
        row = next(r for r in range(ROWS - 1, -1, -1) if state.at(r, action) == 0)
        board = list(state.board)
        board[row * COLS + action] = state.player
        winner = state.player if self._is_winning_move(tuple(board), row, action) else 0
        return ConnectFourState(tuple(board), 3 - state.player, winner, state.moves + 1)

    @staticmethod
    def _is_winning_move(board: tuple[int, ...], row: int, col: int) -> bool:
        player = board[row * COLS + col]
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and board[r * COLS + c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    def is_terminal(self, state: ConnectFourState) -> bool:
        return state.winner != 0 or state.moves == ROWS * COLS

    def get_reward(self, state: ConnectFourState, player: int) -> float:
        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")
        if not self.is_terminal(state) or state.winner == 0:
            return 0.0
        return 1.0 if state.winner == player else -1.0

    def current_player(self, state: ConnectFourState) -> int:
        return state.player

    def render(self, state: ConnectFourState) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(symbols[state.at(r, c)] for c in range(COLS)) for r in range(ROWS)]
        status = f"Winner: {symbols[state.winner]}" if state.winner else f"Turn: {symbols[state.player]}"
        return "0 1 2 3 4 5 6\n" + "\n".join(rows) + f"\n{status}"
