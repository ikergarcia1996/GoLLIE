import ast
import inspect
from dataclasses import dataclass
from typing import Any, List, Union

import black


@dataclass
class APIOutput:
    text: str
    guidelines: List[type]
    annotations: List[object]


class GollieAPI:
    def __init__(self, model: str, base_url: str = None, use_chat_format: bool = True) -> None:
        ...

    def __call__(
        self, texts: Union[List[str], str], guidelines: List[type], **kwargs
    ) -> Union[APIOutput, List[APIOutput]]:
        ...

    @staticmethod
    def _ensure_safe_eval(expression: str, valid_calls: List[str] = []) -> bool:
        """
        Ensures that the given expression is safe to evaluate by parsing and checking it against a list of valid calls.
        Args:
            expression (str): The expression to be evaluated.
            valid_calls (List[str], optional): A list of valid function names that are allowed in the expression. Defaults to an empty list.
        Returns:
            bool: The result of evaluating the expression if it is deemed safe.
        Raises:
            SyntaxError: If the expression contains invalid syntax.
            ValueError: If the expression contains calls that are not in the list of valid calls.
        """

        parsed_expression = ast.parse(expression, mode="eval")
        GollieAPI._check_expression(parsed_expression, valid_calls)

        return eval(ast.unparse(parsed_expression))

    @staticmethod
    def _check_expression(expression: Any, valid_calls: List[str] = [], default: Any = ast.List(elts=[])) -> bool:
        """
        Checks if the given AST expression is safe to evaluate based on the provided valid calls.

        Args:
            expression (Any): The AST expression to check.
            valid_calls (List[str], optional): A list of valid function names that are allowed in the expression. Defaults to an empty list.
            default (Any, optional): The default AST node to replace unsafe expressions with. Defaults to an empty AST List node.

        Returns:
            bool: True if the expression is safe to evaluate, False otherwise.
        """
        if isinstance(expression, ast.Expression):
            if not GollieAPI._check_expression(expression.body, valid_calls):
                expression.body = default
            return True

        elif isinstance(expression, ast.List):
            for i in range(len(expression.elts) - 1, -1, -1):
                if not GollieAPI._check_expression(expression.elts[i], valid_calls):
                    del expression.elts[i]
            return True

        elif isinstance(expression, ast.Call):
            if expression.func.id not in valid_calls:
                return False

            for i in range(len(expression.args) - 1, -1, -1):
                if not GollieAPI._check_expression(expression.args[i], valid_calls):
                    del expression.args[i]

            for i in range(len(expression.keywords) - 1, -1, -1):
                if not GollieAPI._check_expression(expression.keywords[i], valid_calls):
                    del expression.keywords[i]

            return True
        elif isinstance(expression, ast.keyword):
            return GollieAPI._check_expression(expression.value, valid_calls)
        elif isinstance(expression, ast.Constant):
            return True
        else:
            return False

    @staticmethod
    def _prepare_class_definitions(class_definitions: List[type]) -> str:
        """
        Prepares and formats class definitions into a single string.
        This function takes a list of class definitions, de-indents their source code,
        formats them using the `black` code formatter, and joins them into a single string.

        Args:
            class_definitions (List[type]): A list of class types to be formatted.
        Returns:
            str: A single string containing the formatted class definitions.
        Raises:
            RuntimeError: If any of the class definitions are not defined in a file,
                          which is required by `inspect.getsource`.
        """

        def de_indent(code: str) -> str:
            lines = code.split("\n")
            indent = len(lines[0]) - len(lines[0].lstrip())
            return "\n".join([line[indent:] for line in lines])

        try:
            definitions = [
                black.format_str(de_indent(inspect.getsource(definition)), mode=black.Mode())
                for definition in class_definitions
            ]
        except TypeError:
            raise RuntimeError(
                "Class definitions must be defined in a file. See `inspect.getsource` for more information."
            )

        return "\n\n".join(definitions)


if __name__ == "__main__":

    @dataclass
    class Person:
        """This class represents any kind of human being"""

        name: str  # The text referring to the person.
        age: Any = None  # The age

    @dataclass
    class Animal:
        """This class represents any kind of animal"""

        name: str  # The text referring to the person.
        age: Any = None  # The age

    # Check GollieAPI._prepare_class_definitions
    print(GollieAPI._prepare_class_definitions([Person, Animal]))

    # Check GollieAPI._ensure_safe_eval method
    malicious_code = "[Person(name='Oscar'), Organization('Patata'), Person(name='Peter', age=print('patata'))]"
    print("Malicious code:", malicious_code)
    print(
        "Safe eval:",
        GollieAPI._ensure_safe_eval(
            "[Person(name='Oscar'), Organization('Patata'), Person(name='Peter', age=print('patata'))]",
            valid_calls=["Person"],
        ),
    )
