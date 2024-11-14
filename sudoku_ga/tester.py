import random
import subprocess
import numpy as np

def generate_sudoku():
    """Generates a valid Sudoku puzzle with 20-40 filled cells."""
    # Placeholder: Simple generator that creates valid Sudoku puzzles (you can use an external library for better generation)
    from sudoku import Sudoku  # Assuming you have a valid library; adapt this as needed

    puzzle = Sudoku(3).difficulty(0.5).board  # Generates a 9x9 board
    filled_cells = sum(1 for row in puzzle for cell in row if cell != 0)

    # Adjust the number of given numbers if needed (between 20 and 40)
    while filled_cells < 20 or filled_cells > 40:
        puzzle = Sudoku(3).difficulty(0.5).board
        filled_cells = sum(1 for row in puzzle for cell in row if cell != 0)

    return puzzle

def print_sudoku(board):
    """Formats a Sudoku board for display."""
    for row in board:
        print(" ".join(str(num) if num != 0 else "." for num in row))

def run_solver(input_board):
    """Runs the solver from main.cpp on the input board."""
    # Write the input board to a file (or use another IPC mechanism as needed)
    with open("input_sudoku.txt", "w") as file:
        for row in input_board:
            file.write(" ".join(str(num) for num in row) + "\n")

    # Call your C++ solver
    result = subprocess.run(["./solver"], input="input_sudoku.txt", text=True, capture_output=True)

    # Read and parse the output
    output_board = []
    for line in result.stdout.splitlines():
        output_board.append(list(map(int, line.split())))

    return output_board

def validate_solution(board):
    """Validates a Sudoku solution."""
    def is_valid(arr):
        arr = [num for num in arr if num != 0]
        return len(arr) == len(set(arr))

    for row in board:
        if not is_valid(row):
            return False

    for col in zip(*board):
        if not is_valid(col):
            return False

    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = [board[r][c] for r in range(box_row, box_row+3) for c in range(box_col, box_col+3)]
            if not is_valid(box):
                return False

    return True

def main():
    for _ in range(10):  # Number of test cases
        puzzle = generate_sudoku()
        correct_solution = np.array(puzzle)  # Assuming you have a solver library for correct solutions

        print("Generated Sudoku Puzzle:")
        print_sudoku(puzzle)

        solver_output = run_solver(puzzle)
        if not validate_solution(solver_output) or not np.array_equal(correct_solution, solver_output):
            print("\nMismatch detected!")
            print("Original Puzzle:")
            print_sudoku(puzzle)
            print("\nCorrect Solution:")
            print_sudoku(correct_solution)
            print("\nSolver Output:")
            print_sudoku(solver_output)
        else:
            print("Solution is correct.")

if __name__ == "__main__":
    main()
