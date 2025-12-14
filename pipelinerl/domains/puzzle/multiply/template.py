template = """
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class MultiplyGame:
    a: int
    b: int

    def check_answer(self, response: str) -> bool:
        try:
            expression = re.search(r'<answer>(.*?)</answer>', response).group(1)
            print(expression)
            return int(expression) == self.a * self.b
        except Exception:
            return False


def main() -> int:
    a = {a}
    b = {b}
    game = MultiplyGame(a, b)
    answer = input(f"What is {{a}} x {{b}}? Show your work then put your answer in <answer>...</answer> tag: ")

    if game.check_answer(answer):
        print("Correct.")
        return 0
    else:
        print("Wrong.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""



if __name__ == "__main__":
    print(template.format(a=6, b=7))