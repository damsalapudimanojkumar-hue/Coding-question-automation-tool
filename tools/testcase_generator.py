"""
=============================================================
GENERALIZED TEST CASE GENERATOR  (code-editor / function-based questions)
=============================================================
Generates platform-compatible JSON for a coding question given:
  1. A question config (question text, difficulty, etc.)
  2. A reference solution function
  3. A list of test case input definitions

Reproduced into this pipeline from the "Coding Assignment Creation
(Multiple Testcases)" project so Agent 3 (code-editor mode) can run it.
MACHINERY: executed, never injected into an LLM prompt.

NOTE: verify this matches your canonical generator.py exactly; it is
execution-critical. It is validated in this repo by running it on the
BCE example config and checking the outputs.

Usage:
    python tools/testcase_generator.py --config config.json --output out.zip

Programmatic:
    from tools.testcase_generator import TestCaseGenerator
    gen = TestCaseGenerator(config)
    gen.add_test_case(inputs={...}, is_hidden=False, weightage=10)
    gen.generate("output.zip")
=============================================================
"""

import numpy as np
import json
import uuid
import sys
import os
import base64
import argparse
from typing import Any


class _TorchMock:
    """
    Lightweight mock of torch using numpy, for test case generation
    when PyTorch is not installed. Supports basic tensor operations.
    """

    class Tensor:
        def __init__(self, data):
            if isinstance(data, np.ndarray):
                self.data = data.astype(np.float64)
            else:
                self.data = np.array(data, dtype=np.float64)

        @property
        def T(self):
            return _TorchMock.Tensor(self.data.T)

        def __matmul__(self, other):
            if isinstance(other, _TorchMock.Tensor):
                return _TorchMock.Tensor(self.data @ other.data)
            return _TorchMock.Tensor(self.data @ np.array(other))

        def __add__(self, other):
            if isinstance(other, _TorchMock.Tensor):
                return _TorchMock.Tensor(self.data + other.data)
            return _TorchMock.Tensor(self.data + np.array(other))

        def __radd__(self, other):
            return self.__add__(other)

        def __sub__(self, other):
            if isinstance(other, _TorchMock.Tensor):
                return _TorchMock.Tensor(self.data - other.data)
            return _TorchMock.Tensor(self.data - np.array(other))

        def __rsub__(self, other):
            if isinstance(other, _TorchMock.Tensor):
                return _TorchMock.Tensor(other.data - self.data)
            return _TorchMock.Tensor(np.array(other) - self.data)

        def __mul__(self, other):
            if isinstance(other, _TorchMock.Tensor):
                return _TorchMock.Tensor(self.data * other.data)
            return _TorchMock.Tensor(self.data * other)

        def __rmul__(self, other):
            return self.__mul__(other)

        def __truediv__(self, other):
            if isinstance(other, _TorchMock.Tensor):
                return _TorchMock.Tensor(self.data / other.data)
            return _TorchMock.Tensor(self.data / other)

        def __neg__(self):
            return _TorchMock.Tensor(-self.data)

        def __pow__(self, other):
            return _TorchMock.Tensor(self.data ** other)

        def sum(self, dim=None, keepdim=False):
            return _TorchMock.Tensor(self.data.sum(axis=dim, keepdims=keepdim))

        def mean(self, dim=None, keepdim=False):
            return _TorchMock.Tensor(self.data.mean(axis=dim, keepdims=keepdim))

        def view(self, *shape):
            return _TorchMock.Tensor(self.data.reshape(*shape))

        def reshape(self, *shape):
            return _TorchMock.Tensor(self.data.reshape(*shape))

        def unsqueeze(self, dim):
            return _TorchMock.Tensor(np.expand_dims(self.data, axis=dim))

        def squeeze(self, dim=None):
            return _TorchMock.Tensor(np.squeeze(self.data, axis=dim))

        def item(self):
            return self.data.item()

        def tolist(self):
            return self.data.tolist()

        def numpy(self):
            return self.data

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return _TorchMock.Tensor(self.data.astype(np.float64))

        def __repr__(self):
            return f"MockTensor({self.data})"

        @property
        def shape(self):
            return self.data.shape

    float32 = "float32"
    float64 = "float64"
    int64 = "int64"

    @staticmethod
    def tensor(data, dtype=None):
        return _TorchMock.Tensor(data)

    @staticmethod
    def zeros(*shape, dtype=None):
        return _TorchMock.Tensor(np.zeros(shape))

    @staticmethod
    def ones(*shape, dtype=None):
        return _TorchMock.Tensor(np.ones(shape))

    @staticmethod
    def randn(*shape):
        return _TorchMock.Tensor(np.random.randn(*shape))

    @staticmethod
    def exp(t):
        return _TorchMock.Tensor(np.exp(t.data))

    @staticmethod
    def log(t):
        return _TorchMock.Tensor(np.log(t.data))

    @staticmethod
    def relu(t):
        return _TorchMock.Tensor(np.maximum(0, t.data))

    @staticmethod
    def sigmoid(t):
        return _TorchMock.Tensor(1 / (1 + np.exp(-t.data)))

    @staticmethod
    def tanh(t):
        return _TorchMock.Tensor(np.tanh(t.data))

    @staticmethod
    def softmax(t, dim=-1):
        e = np.exp(t.data - np.max(t.data, axis=dim, keepdims=True))
        return _TorchMock.Tensor(e / e.sum(axis=dim, keepdims=True))

    @staticmethod
    def matmul(a, b):
        return _TorchMock.Tensor(a.data @ b.data)

    @staticmethod
    def cat(tensors, dim=0):
        return _TorchMock.Tensor(np.concatenate([t.data for t in tensors], axis=dim))

    @staticmethod
    def stack(tensors, dim=0):
        return _TorchMock.Tensor(np.stack([t.data for t in tensors], axis=dim))

    @staticmethod
    def clamp(t, min=None, max=None):
        return _TorchMock.Tensor(np.clip(t.data, min, max))

    @staticmethod
    def sqrt(t):
        return _TorchMock.Tensor(np.sqrt(t.data))

    @staticmethod
    def abs(t):
        return _TorchMock.Tensor(np.abs(t.data))

    @staticmethod
    def max(t, dim=None, keepdim=False):
        if dim is None:
            return _TorchMock.Tensor(np.array(np.max(t.data)))
        vals = np.max(t.data, axis=dim, keepdims=keepdim)
        idxs = np.argmax(t.data, axis=dim)
        return _TorchMock.Tensor(vals), _TorchMock.Tensor(idxs)

    @staticmethod
    def min(t, dim=None, keepdim=False):
        if dim is None:
            return _TorchMock.Tensor(np.array(np.min(t.data)))
        vals = np.min(t.data, axis=dim, keepdims=keepdim)
        idxs = np.argmin(t.data, axis=dim)
        return _TorchMock.Tensor(vals), _TorchMock.Tensor(idxs)

    class nn:
        @staticmethod
        def functional():
            pass

        class functional:
            @staticmethod
            def relu(t):
                return _TorchMock.relu(t)

            @staticmethod
            def sigmoid(t):
                return _TorchMock.sigmoid(t)

            @staticmethod
            def softmax(t, dim=-1):
                return _TorchMock.softmax(t, dim)

            @staticmethod
            def cross_entropy(input, target):
                log_softmax = input.data - np.log(
                    np.exp(input.data).sum(axis=-1, keepdims=True)
                )
                n = input.data.shape[0]
                loss = -log_softmax[np.arange(n), target.data.astype(int)].mean()
                return _TorchMock.Tensor(np.array(loss))


class TestCaseGenerator:
    def __init__(self, config: dict):
        """
        Initialize with a config dict containing:
        - question_text: str (markdown description)
        - short_text: str (title)
        - difficulty: str (EASY/MEDIUM/HARD)
        - function_name: str (name of the function to test)
        - param_names: list[str] (ordered parameter names)
        - starter_code: str (code shown to user)
        - solution_code: str (reference solution code)
        - language: str (default: PYTHON38_DATASCIENCE)
        - time_limit: float (default: 4.0)
        - rounding: int (decimal places, default: 4)
        """
        self.config = config
        self.test_cases = []
        self.solution_func = None
        self._load_solution()

    def _load_solution(self):
        """Dynamically load the solution function from solution_code string"""
        solution_code = self.config["solution_code"]
        func_name = self.config["function_name"]
        library = self.config.get("library", "numpy")

        # Create a temporary module namespace
        namespace = {"np": np, "numpy": np}

        # Try importing torch, fall back to mock if not available
        if library == "pytorch":
            try:
                import torch
                namespace["torch"] = torch
            except ImportError:
                # Inject mock into sys.modules so 'import torch' works in exec
                mock = _TorchMock()
                sys.modules["torch"] = mock
                namespace["torch"] = mock

        exec(solution_code, namespace)

        # Clean up mock from sys.modules
        if library == "pytorch" and "torch" in sys.modules and isinstance(sys.modules["torch"], _TorchMock):
            del sys.modules["torch"]

        if func_name not in namespace:
            raise ValueError(
                f"Function '{func_name}' not found in solution code. "
                f"Available: {[k for k in namespace if not k.startswith('_')]}"
            )

        self.solution_func = namespace[func_name]

    def _run_solution(self, inputs: dict) -> Any:
        """Run the reference solution with given inputs"""
        param_names = self.config["param_names"]
        library = self.config.get("library", "numpy")
        args = [inputs[name] for name in param_names]

        # Convert lists to appropriate array type
        converted_args = []
        for arg in args:
            if isinstance(arg, list):
                if library == "pytorch":
                    try:
                        import torch
                        converted_args.append(torch.tensor(arg, dtype=torch.float32))
                    except ImportError:
                        converted_args.append(np.array(arg, dtype=np.float64))
                elif library == "pandas":
                    # A list of record dicts becomes a DataFrame; the solution owns
                    # any dtype coercion (e.g. pd.to_datetime on a date column).
                    import pandas as pd
                    converted_args.append(pd.DataFrame(arg))
                else:
                    converted_args.append(np.array(arg))
            else:
                converted_args.append(arg)

        result = self.solution_func(*converted_args)
        return result

    def _format_output(self, result: Any) -> str:
        """Format the output to a rounded string representation"""
        rounding = self.config.get("rounding", 4)

        # pandas questions: the expected output is the DataFrame/Series printed
        # WITHOUT its index (matches the platform's grading representation).
        try:
            import pandas as pd
            if isinstance(result, (pd.DataFrame, pd.Series)):
                return result.to_string(index=False)
        except ImportError:
            pass

        # Handle torch tensors (real or mock)
        if hasattr(result, 'detach') and hasattr(result, 'numpy'):
            result = result.detach().cpu().numpy()
        if hasattr(result, 'data') and isinstance(result.data, np.ndarray):
            result = result.data

        if isinstance(result, np.ndarray):
            result = result.tolist()

        def round_recursive(val):
            if isinstance(val, float):
                return round(val, rounding)
            if isinstance(val, list):
                return [round_recursive(x) for x in val]
            return val

        result = round_recursive(result)

        return str(result)

    def _format_input(self, inputs: dict) -> str:
        """Format inputs as JSON string in list format [arg1, arg2, ...]"""
        param_names = self.config["param_names"]
        input_list = []

        for name in param_names:
            val = inputs[name]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            input_list.append(val)

        return json.dumps(input_list)

    def add_test_case(
        self,
        inputs: dict,
        is_hidden: bool = False,
        weightage: int = 10,
        tags: list = None,
        display_text: str = None,
    ):
        """
        Add a single test case.

        Args:
            inputs: dict mapping param_name -> value
            is_hidden: whether this test case is hidden from the user
            weightage: score weight for this test case
            tags: optional tags
            display_text: optional display text
        """
        # Validate that all required params are provided
        param_names = self.config["param_names"]
        missing = [p for p in param_names if p not in inputs]
        if missing:
            raise ValueError(f"Missing input parameters: {missing}")

        # Run solution to get expected output
        result = self._run_solution(inputs)
        output_str = self._format_output(result)
        input_str = self._format_input(inputs)

        order = len(self.test_cases) + 1

        self.test_cases.append({
            "id": str(uuid.uuid4()),
            "input": input_str,
            "output": output_str,
            "is_hidden": is_hidden,
            "weightage": weightage,
            "evaluation_type": "DEFAULT",
            "display_text": display_text,
            "criteria": None,
            "tags": tags or [],
            "order": order,
        })

    def add_test_cases_bulk(self, test_definitions: list):
        """Add multiple test cases at once. Weightages are auto-normalized so they
        always sum to exactly 100, distributed as evenly as possible across however
        many cases there are (e.g. 20 -> 5 each; 15 -> ten 7s + five 6s)."""
        tds = normalize_weightages(list(test_definitions))
        for td in tds:
            self.add_test_case(
                inputs=td["inputs"],
                is_hidden=td.get("is_hidden", False),
                weightage=td.get("weightage", 10),
                tags=td.get("tags"),
                display_text=td.get("display_text"),
            )

    def _generate_evaluation_code(self) -> str:
        """
        Generate universal main.py evaluation code.
        Only the FUNC_NAME changes per question -- everything else is the same.
        Handles dict, list, and single value inputs automatically.
        Supports both NumPy and PyTorch outputs.
        """
        func_name = self.config["function_name"]
        library = self.config.get("library", "numpy")  # "numpy" or "pytorch"

        if library == "pytorch":
            eval_code = f"""import json
import inspect
import numpy as np
import torch
import solution
import importlib

# Reload solution (useful in notebooks)
importlib.reload(solution)

# ============================
# CONFIG (per question)
# ============================
FUNC_NAME = "{func_name}"   # function under test


# ============================
# Utilities
# ============================

def safe_json_loads(s):
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    data = json.loads(s)
    if isinstance(data, str):
        data = json.loads(data)
    return data


def should_convert_to_tensor(value):
    if not isinstance(value, list) or not value:
        return False
    if isinstance(value[0], list):
        return True
    return all(isinstance(x, (int, float)) for x in value)


def convert_value(value):
    if should_convert_to_tensor(value):
        return torch.tensor(value, dtype=torch.float32)
    if isinstance(value, (int, float)):
        return torch.tensor(value, dtype=torch.float32)
    return value


def normalize_output(result):
    if isinstance(result, torch.Tensor):
        result = result.detach().cpu().numpy().tolist()
    if isinstance(result, np.ndarray):
        result = result.tolist()
    if isinstance(result, tuple):
        return tuple(normalize_output(r) for r in result)
    if isinstance(result, dict):
        return {{k: normalize_output(v) for k, v in result.items()}}
    def round_if_float(x):
        return round(x, 4) if isinstance(x, float) else x
    if isinstance(result, list):
        if result and isinstance(result[0], list):
            return [[round_if_float(x) for x in row] for row in result]
        return [round_if_float(x) for x in result]
    if isinstance(result, float):
        return round(result, 4)
    return result


# ============================
# Main Evaluator
# ============================

def main():
    input_str = input().strip()
    data = safe_json_loads(input_str)

    func = getattr(solution, FUNC_NAME)
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    if isinstance(data, dict):
        kwargs = {{}}
        for name in param_names:
            if name in data:
                kwargs[name] = convert_value(data[name])
        result = func(**kwargs)
    elif isinstance(data, list):
        args = [convert_value(v) for v in data]
        result = func(*args)
    else:
        result = func(convert_value(data))

    result = normalize_output(result)
    print(str(result))


if __name__ == "__main__":
    main()
"""
        else:
            eval_code = f"""import json
import inspect
import numpy as np
import solution
import importlib

# Reload solution (useful in notebooks)
importlib.reload(solution)

# ============================
# CONFIG (per question)
# ============================
FUNC_NAME = "{func_name}"   # function under test


# ============================
# Utilities
# ============================

def safe_json_loads(s):
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    data = json.loads(s)
    if isinstance(data, str):
        data = json.loads(data)
    return data


def should_convert_to_numpy(value):
    if not isinstance(value, list) or not value:
        return False
    if isinstance(value[0], list):
        return True
    return all(isinstance(x, (int, float)) for x in value)


def convert_value(value):
    if should_convert_to_numpy(value):
        return np.array(value)
    return value


def normalize_output(result):
    if isinstance(result, np.ndarray):
        result = result.tolist()
    if isinstance(result, tuple):
        return tuple(normalize_output(r) for r in result)
    def round_if_float(x):
        return round(x, 4) if isinstance(x, float) else x
    if isinstance(result, list):
        if result and isinstance(result[0], list):
            return [[round_if_float(x) for x in row] for row in result]
        return [round_if_float(x) for x in result]
    if isinstance(result, float):
        return round(result, 4)
    return result


# ============================
# Main Evaluator
# ============================

def main():
    input_str = input().strip()
    data = safe_json_loads(input_str)

    func = getattr(solution, FUNC_NAME)
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    if isinstance(data, dict):
        kwargs = {{}}
        for name in param_names:
            if name in data:
                kwargs[name] = convert_value(data[name])
        result = func(**kwargs)
    elif isinstance(data, list):
        args = [convert_value(v) for v in data]
        result = func(*args)
    else:
        result = func(convert_value(data))

    result = normalize_output(result)
    print(str(result))


if __name__ == "__main__":
    main()
"""
        return eval_code

    def _encode_base64(self, text: str) -> str:
        """Encode text to base64 string"""
        return base64.b64encode(text.encode("utf-8")).decode("utf-8")

    def _build_question_object(self) -> tuple:
        """Build one question object + its question_id (factored out so multiple
        questions can be combined into a single bundle)."""
        total_score = sum(tc["weightage"] for tc in self.test_cases)
        language = self.config.get("language", "PYTHON38_DATASCIENCE")

        eval_code = self._generate_evaluation_code()
        eval_code_base64 = self._encode_base64(eval_code)

        question_text = self.config.get("rephrased_question_text", self.config["question_text"])
        short_text = self.config.get("rephrased_short_text", self.config["short_text"])

        question_id = str(uuid.uuid4())

        question_object = {
            "test_cases": self.test_cases,
            "total_score": total_score,
            "question_type": "CODING",
            "question_asked_by_companies_info": [],
            "question": {
                "difficulty": self.config.get("difficulty", "EASY"),
                "content": question_text,
                "short_text": short_text,
                "multimedia": [],
                "language": "ENGLISH",
                "content_type": "MARKDOWN",
                "question_id": question_id,
                "default_tag_names": [self.config.get("difficulty", "EASY")],
                "concept_tag_names": [],
                "metadata": None,
            },
            "coding_question_details": [
                {
                    "code_content": self.config["starter_code"],
                    "default_code": True,
                    "language": language,
                    "code_id": str(uuid.uuid4()),
                }
            ],
            "code_repository_details": None,
            "language_code_repository_details": [
                {
                    "language": language,
                    "file_path_to_execute": "main.py",
                    "default_file_path_to_submit_code": "solution.py",
                    "code_repository": [
                        {
                            "file_name": "main.py",
                            "file_type": "FILE",
                            "file_content": eval_code_base64,
                        }
                    ],
                }
            ],
            "solutions": [
                {
                    "order": 1,
                    "title": {"content": "Code", "content_type": "TEXT"},
                    "description": {"content": "", "content_type": ""},
                    "code_details": [
                        {
                            "code_content": self.config["solution_code"],
                            "language": language,
                            "default_code": True,
                            "code_id": str(uuid.uuid4()),
                        }
                    ],
                    "complexity_analysis": {"content": "", "content_type": ""},
                }
            ],
            "hints": [],
            "test_case_evaluation_metrics": [
                {
                    "language": language,
                    "time_limit_to_execute_in_seconds": self.config.get("time_limit", 4.0),
                }
            ],
        }
        return question_object, question_id

    def generate(self, output_path: str = None) -> list:
        """
        Generate the full JSON structure, question_sets_questions JSON,
        and package both into a zip file. Returns the full JSON list.
        """
        import zipfile

        if not self.test_cases:
            raise ValueError("No test cases added. Use add_test_case() first.")

        question_object, question_id = self._build_question_object()
        full_json = [question_object]
        question_sets_json = [
            {
                "question_set_id": str(uuid.uuid4()),
                "question_id": question_id,
                "order": 1,
            }
        ]

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            zip_path = output_path if output_path.endswith(".zip") else os.path.splitext(output_path)[0] + ".zip"

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("questions.json", json.dumps(full_json, indent=2))
                zf.writestr("question_sets_questions.json", json.dumps(question_sets_json, indent=2))

            total_score = question_object["total_score"]
            print(f"[OK] Generated {len(self.test_cases)} test cases -> {zip_path}")
            print(f"   Total score: {total_score}")
            print(
                f"   Visible: {sum(1 for tc in self.test_cases if not tc['is_hidden'])}, "
                f"Hidden: {sum(1 for tc in self.test_cases if tc['is_hidden'])}"
            )
            print(f"   Question ID: {question_id}")
            print(f"   Zip contains: questions.json, question_sets_questions.json")

        return full_json

    def preview(self):
        """Print a preview of all test cases"""
        print(f"\n{'='*60}")
        print(f"  {self.config['short_text']}")
        print(f"  Difficulty: {self.config.get('difficulty', 'EASY')}")
        print(f"  Function: {self.config['function_name']}")
        print(f"  Params: {self.config['param_names']}")
        print(f"{'='*60}\n")

        for tc in self.test_cases:
            hidden_tag = " [HIDDEN]" if tc["is_hidden"] else ""
            print(f"  Test #{tc['order']}{hidden_tag} (weight: {tc['weightage']})")
            print(f"    Input:  {tc['input']}")
            print(f"    Output: {tc['output']}")
            print()


def normalize_weightages(test_definitions: list) -> list:
    """Redistribute weightages so they sum to EXACTLY 100, as evenly as possible,
    regardless of how many test cases there are. Integer weights; the first few cases
    absorb the remainder. Examples: 20 cases -> 5 each; 15 -> ten 7s + five 6s (=100);
    8 -> four 13s + four 12s (=100). Mutates and returns the same list."""
    n = len(test_definitions)
    if n == 0:
        return test_definitions
    base = 100 // n
    remainder = 100 - base * n
    for i, td in enumerate(test_definitions):
        td["weightage"] = base + (1 if i < remainder else 0)
    return test_definitions


def generate_bundle(configs: list, output_path: str) -> dict:
    """
    Batch mode: multiple questions in ONE bundle (same two JSON files).
    `configs` is a list of full config dicts (each with test_definitions).
    questions.json becomes an array of N question objects; the set file gets
    N mappings (order 1..N). Each question keeps its own question_id + evaluator.
    """
    import zipfile

    questions = []
    set_maps = []
    set_id = str(uuid.uuid4())
    for i, config in enumerate(configs, 1):
        test_definitions = config.pop("test_definitions", [])
        gen = TestCaseGenerator(config)
        if test_definitions:
            gen.add_test_cases_bulk(test_definitions)
        obj, qid = gen._build_question_object()
        questions.append(obj)
        set_maps.append({"question_set_id": set_id, "question_id": qid, "order": i})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    zip_path = output_path if output_path.endswith(".zip") else os.path.splitext(output_path)[0] + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("questions.json", json.dumps(questions, indent=2))
        zf.writestr("question_sets_questions.json", json.dumps(set_maps, indent=2))
    print(f"[OK] Generated bundle of {len(questions)} questions -> {zip_path}")
    return {"questions": questions, "question_sets": set_maps}


def load_config_from_json(config_path: str) -> dict:
    """Load a config JSON file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate test cases from config")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--output", default=None, help="Output zip file path")
    parser.add_argument("--preview", action="store_true", help="Preview test cases")
    args = parser.parse_args()

    config = load_config_from_json(args.config)

    # Separate test_definitions from config
    test_definitions = config.pop("test_definitions", [])

    gen = TestCaseGenerator(config)

    if test_definitions:
        gen.add_test_cases_bulk(test_definitions)

    if args.preview:
        gen.preview()

    output_path = args.output or config.get("output_path", "output_testcases.zip")
    gen.generate(output_path)


if __name__ == "__main__":
    main()
