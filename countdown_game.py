import re
from collections import Counter


class CountdownGame:
    def __init__(self, numbers: list[int], target: int):
        self.numbers = numbers
        self.target = target

    def check_answer(self, answer: str) -> bool:
        """check if the answer has the correct format and is correct"""
        try:
            expression = answer.split("<answer>")[1].split("</answer>")[0].strip()
        except Exception:
            return False
        
        # Validate format
        if not re.match(r'^[\d\s\+\-\*\/\(\)]+$', expression):
            return False

        # Extract and validate numbers
        used_numbers = [int(x) for x in re.findall(r'\d+', expression)]
        if Counter(used_numbers) != Counter(self.numbers):
            return False

        # Evaluate and check result
        try:
            result = eval(expression)
            return result == self.target
        except Exception:
            return False


if __name__ == "__main__":
    # Create a countdown game
    game = CountdownGame([1, 2, 3, 4], 10)
    success = game.check_answer(input("Enter expression in <answer>...</answer> format: "))
    if success:
        print("Success!")
        exit(0)
    else:
        print("Wrong!")
        exit(1)


# python countdown_game.py 
#Enter expression (or <answer>...</answer>):
