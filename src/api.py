import ast
import inspect
import json
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import black
import torch
from jinja2 import Template

from src.model import load_model
from src.tasks.utils_data import SYSTEM_PROMPT
from transformers import AutoConfig


DEFAULT_USER_TEMPLATE = Template(
    """Extract the relevant information from the following text:
```python
text = {{ text.__repr__() }}
```
"""
)


@dataclass()
class InputExample:
    text: str
    annotations: str


@dataclass
class APIOutput:
    text: str
    guidelines: List[type]
    annotations: Union[List[object], List[List[object]]]


def prepare_data_for_chat(texts: List[str], guidelines: str, examples: List[InputExample] = None):
    base_conversation = []

    base_conversation.append(SYSTEM_PROMPT.format(guidelines=guidelines))
    for example in examples:
        base_conversation.append(
            {
                "role": "user",
                "content": DEFAULT_USER_TEMPLATE.render(text=example.text),
            }
        )
        base_conversation.append(
            {
                "role": "assistant",
                "content": "```python\n" + example.annotations + "\n```",
            }
        )

    conversations = []
    for text in texts:
        conversations.append(
            base_conversation
            + [
                {
                    "role": "user",
                    "content": DEFAULT_USER_TEMPLATE.render(text=text),
                }
            ]
        )


def prepare_data(texts: List[str], guidelines: str):
    ...


class GollieInferenceWrapper:
    def __init__(self) -> None:
        return NotImplementedError

    def __call__(
        self,
        texts: Union[List[str], str],
        guidelines: List[type],
        examples: List[Tuple[str, List[object]]] = None,
        **kwargs,
    ) -> Union[APIOutput, List[APIOutput]]:
        raise NotImplementedError


class LocalGollieInferenceWrapper(GollieInferenceWrapper):
    def __init__(
        self,
        model: str,
        use_chat_format: bool = True,
        dtype: str = "bfloat16",
        use_flash_attention: bool = True,
        **kwargs,
    ) -> None:
        try:
            self.config = AutoConfig.from_pretrained(model)
            model_name_or_path = model
            use_lora = False
        except OSError as e:
            msg = e.__str__()
            if "does not appear to have a file named config.json." in msg:
                with open(f"{model}/adapter_config.json", "r") as f:
                    model_name_or_path = json.load(f)["model_name_or_path"]
                    self.config = AutoConfig.from_pretrained(model_name_or_path)
                    use_lora = True
            else:
                raise e

        self.lora_name_or_path = model if use_lora else None
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.use_chat_format = use_chat_format

        self.model, self.tokenizer = load_model(
            inference=True,
            model_weights_name_or_path=model_name_or_path,
            use_lora=use_lora,
            lora_weights_name_or_path=self.lora_name_or_path,
            torch_dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            force_auto_device_map=True,
        )

    def __call__(
        self,
        texts: Union[List[str], str],
        guidelines: str,
        examples: List[Tuple[str, str]] = None,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> Union[List[str], List[List[str]]]:
        if not self.use_chat_format and examples is not None:
            raise ValueError("Examples are only supported when using chat format.")

        if not isinstance(texts, list):
            texts = [texts]

        if self.use_chat_format:
            input_data = prepare_data_for_chat(texts, guidelines, examples)
            input_data = self.tokenizer.apply_chat_template(
                input_data, add_generation_prompt=True, return_tensors="pt", padding=True
            )
        else:
            input_data = prepare_data(texts, guidelines)
            input_data = self.tokenizer(input_data, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_data.input_ids,
                return_dict_in_generate=True,
                num_return_sequences=num_return_sequences,
                **kwargs,
            ).sequences

        outputs: List[str] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(len(outputs)):
            if self.use_chat_format:
                outputs[i] = outputs[i].strip().split("```python\n")[-1].rstrip("```")
            else:
                outputs[i] = outputs[i].strip().strip().split("result = ")[-1]

        if num_return_sequences > 1:
            output_list = []
            for i in range(len(texts)):
                output_list.append(outputs[i * num_return_sequences : (i + 1) * num_return_sequences])

        return outputs


class RemoteGollieInferenceWrapper(GollieInferenceWrapper):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __call__(
        self,
        texts: Union[List[str], str],
        guidelines: List[type],
        examples: List[Tuple[str, str]] = None,
        **kwargs,
    ) -> Union[List[str], List[List[str]]]:
        raise NotImplementedError


class GollieAPI:
    def __init__(self, model: str, base_url: str = None, use_chat_format: bool = True, **kwargs) -> None:
        if base_url is not None:
            self.inference = RemoteGollieInferenceWrapper(base_url)
        else:
            self.inference = LocalGollieInferenceWrapper(model, use_chat_format, **kwargs)

    def __call__(
        self, texts: Union[List[str], str], guidelines: List[type], **kwargs
    ) -> Union[APIOutput, List[APIOutput]]:
        raise NotImplementedError

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
