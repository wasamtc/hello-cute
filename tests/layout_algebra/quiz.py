#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any, Callable, List, Optional, Sequence


@dataclass(frozen=True)
class LayoutSpec:
    shape: Any
    stride: Any


@dataclass(frozen=True)
class Question:
    title: str
    op_name: str
    layout_a: LayoutSpec
    layout_b: LayoutSpec


@dataclass(frozen=True)
class Backend:
    name: str
    reason: Optional[str]
    make_layout: Callable[[LayoutSpec], Any]
    logical_divide: Callable[[Any, Any], Any]
    logical_product: Callable[[Any, Any], Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _product(value: Any) -> int:
    if isinstance(value, tuple):
        result = 1
        for item in value:
            result *= _product(item)
        return result
    return int(value)


def _prefix_product(shape: Any, init: int = 1):
    if isinstance(shape, tuple):
        result = []
        current = init
        for value in shape:
            result.append(_prefix_product(value, current))
            current *= _product(value)
        return tuple(result)
    return init


def _format_value(value: Any) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(_format_value(item) for item in value) + ")"
    return str(value)


def _format_layout(layout: LayoutSpec) -> str:
    return f"{_format_value(layout.shape)}:{_format_value(layout.stride)}"


def _normalize_layout_answer(text: str) -> str:
    translation = str.maketrans({
        "，": ",",
        "：": ":",
        "（": "(",
        "）": ")",
    })
    return "".join(text.translate(translation).split())


def _random_seed() -> int:
    return random.SystemRandom().randrange(1, 10**12)


def _make_rank1_layout(rng: random.Random) -> LayoutSpec:
    size = rng.randint(5, 12)
    stride = rng.randint(1, 4)
    return LayoutSpec(shape=size, stride=stride)


def _make_rank2_layout(rng: random.Random) -> LayoutSpec:
    rows = rng.randint(2, 4)
    cols = rng.randint(3, 6)

    if rng.choice([True, False]):
        stride = (1, rows)
    else:
        pad = rng.randint(0, 2)
        stride = (cols + pad, 1)

    return LayoutSpec(shape=(rows, cols), stride=stride)


def _make_random_layout(rng: random.Random) -> LayoutSpec:
    if rng.choice([True, False]):
        return _make_rank1_layout(rng)
    return _make_rank2_layout(rng)


def _make_divisor_layout(rng: random.Random, max_size: int) -> LayoutSpec:
    max_tile_size = min(max_size - 1, 6)
    if rng.choice([True, False]):
        size = rng.randint(2, max_tile_size)
        return LayoutSpec(shape=size, stride=1)

    while True:
        rows = rng.randint(1, min(3, max_tile_size))
        cols_upper = min(4, max_tile_size)
        cols = rng.randint(2, cols_upper)
        if rows * cols < max_size:
            return LayoutSpec(shape=(rows, cols), stride=(1, rows))


def _is_valid_operation(op_name: str, layout_a: LayoutSpec, layout_b: LayoutSpec) -> bool:
    python_root = _repo_root() / "third_party" / "cutlass" / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

    import pycute  # type: ignore

    a = pycute.Layout(layout_a.shape, layout_a.stride)
    b = pycute.Layout(layout_b.shape, layout_b.stride)

    try:
        if op_name == "logical_divide":
            pycute.logical_divide(a, b)
        elif op_name == "logical_product":
            pycute.logical_product(a, b)
        else:
            raise ValueError(f"未知操作: {op_name}")
    except Exception:
        return False

    # These operators are built on composition/complement; if pycute can
    # evaluate them on the generated layouts, the combination satisfies the
    # effective preconditions of this quiz.
    return True


def _generate_divide_question(rng: random.Random) -> Question:
    while True:
        layout_a = _make_random_layout(rng)
        layout_b = _make_divisor_layout(rng, _product(layout_a.shape))
        if (
            _product(layout_b.shape) < _product(layout_a.shape)
            and _is_valid_operation("logical_divide", layout_a, layout_b)
        ):
            return Question(
                title="计算 `logical_divide(A, B)` 的结果 layout",
                op_name="logical_divide",
                layout_a=layout_a,
                layout_b=layout_b,
            )


def _generate_product_question(rng: random.Random) -> Question:
    while True:
        layout_a = _make_random_layout(rng)
        layout_b = _make_rank1_layout(rng)
        if _is_valid_operation("logical_product", layout_a, layout_b):
            return Question(
                title="计算 `logical_product(A, B)` 的结果 layout",
                op_name="logical_product",
                layout_a=layout_a,
                layout_b=layout_b,
            )


def _generate_questions(rng: random.Random) -> List[Question]:
    return [
        _generate_divide_question(rng),
        _generate_product_question(rng),
    ]


def _load_backend() -> Backend:
    try:
        import cutlass.cute as cute  # type: ignore

        def make_layout(layout: LayoutSpec):
            return cute.make_layout(layout.shape, stride=layout.stride)

        def logical_divide(layout_a: Any, layout_b: Any):
            return cute.logical_divide(layout_a, layout_b)

        def logical_product(layout_a: Any, layout_b: Any):
            return cute.logical_product(layout_a, layout_b)

        probe_a = make_layout(LayoutSpec(shape=8, stride=1))
        probe_b = make_layout(LayoutSpec(shape=4, stride=1))
        _ = logical_divide(probe_a, probe_b)
        _ = logical_product(probe_a, probe_b)

        return Backend(
            name="cutlass.cute",
            reason=None,
            make_layout=make_layout,
            logical_divide=logical_divide,
            logical_product=logical_product,
        )
    except Exception as exc:
        python_root = _repo_root() / "third_party" / "cutlass" / "python"
        if str(python_root) not in sys.path:
            sys.path.insert(0, str(python_root))

        import pycute  # type: ignore

        def make_layout(layout: LayoutSpec):
            return pycute.Layout(layout.shape, layout.stride)

        def logical_divide(layout_a: Any, layout_b: Any):
            return pycute.logical_divide(layout_a, layout_b)

        def logical_product(layout_a: Any, layout_b: Any):
            return pycute.logical_product(layout_a, layout_b)

        return Backend(
            name="pycute",
            reason=f"{type(exc).__name__}: {exc}",
            make_layout=make_layout,
            logical_divide=logical_divide,
            logical_product=logical_product,
        )


def _compute_answers(backend: Backend, questions: Sequence[Question]) -> List[str]:
    answers: List[str] = []
    for question in questions:
        layout_a = backend.make_layout(question.layout_a)
        layout_b = backend.make_layout(question.layout_b)
        if question.op_name == "logical_divide":
            result = backend.logical_divide(layout_a, layout_b)
        elif question.op_name == "logical_product":
            result = backend.logical_product(layout_a, layout_b)
        else:
            raise ValueError(f"未知操作: {question.op_name}")
        answers.append(str(result))
    return answers


def _print_questions(questions: Sequence[Question]) -> None:
    print("请写出下面每个 layout 运算的结果。")
    print("答案请按 `shape:stride` 的形式输入完整结果，空格不敏感。")
    print("交互模式下每题输入一行。")
    print()
    for index, question in enumerate(questions, start=1):
        print(f"Q{index}. {question.title}")
        print(f"    A = {_format_layout(question.layout_a)}")
        print(f"    B = {_format_layout(question.layout_b)}")
        print()


def _parse_answers(raw_answers: Sequence[str], expected_count: int) -> List[str]:
    answers = [answer for answer in raw_answers if answer.strip()]
    if len(answers) != expected_count:
        raise ValueError(
            f"需要 {expected_count} 个答案，实际收到 {len(answers)} 个。"
        )
    return answers


def _prompt_for_answers(questions: Sequence[Question]) -> List[str]:
    answers: List[str] = []
    for index in range(1, len(questions) + 1):
        answers.append(input(f"请输入 Q{index} 的结果 layout: ").strip())
    return _parse_answers(answers, len(questions))


def _print_result(
    questions: Sequence[Question],
    user_answers: Sequence[str],
    expected_answers: Sequence[str],
) -> int:
    correct = 0
    print()
    print("结果:")
    for index, (question, user_answer, expected_answer) in enumerate(
        zip(questions, user_answers, expected_answers),
        start=1,
    ):
        if _normalize_layout_answer(user_answer) == _normalize_layout_answer(expected_answer):
            correct += 1
            print(f"Q{index} 正确: {expected_answer}")
        else:
            print(f"Q{index} 错误:")
            print(f"    你的答案: {user_answer}")
            print(f"    正确答案: {expected_answer}")
            print(f"    对应题目: {question.op_name}(A, B)")

    print()
    print(f"总分: {correct}/{len(questions)}")
    return 0 if correct == len(questions) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="layout_algebra 交互测试"
    )
    parser.add_argument(
        "--answers",
        nargs="*",
        default=None,
        help="直接提供所有答案，便于非交互运行",
    )
    parser.add_argument(
        "--show-answers",
        action="store_true",
        help="打印标准答案后退出",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="指定随机种子，便于复现同一套题目",
    )
    args = parser.parse_args(argv)

    backend = _load_backend()
    seed = args.seed if args.seed is not None else _random_seed()
    rng = random.Random(seed)
    questions = _generate_questions(rng)

    print(f"当前后端: {backend.name}")
    if backend.reason:
        print(
            "官方 cutlass.cute 在当前环境下不能直接用于原生 Python 的 layout 代数求值，"
            "已回退到仓库内置的 pycute。"
        )
        print(f"回退原因: {backend.reason}")
        print(
            "这是 CuTe DSL 当前的使用限制: layout algebra 主要支持在 `@cute.jit` "
            "函数内部执行；本测试是命令行交互题，因此判分阶段默认使用等价的 "
            "`pycute` 镜像实现。"
        )
        print()

    print(f"题目 seed: {seed}")
    print()

    expected_answers = _compute_answers(backend, questions)
    _print_questions(questions)

    if args.show_answers:
        for index, answer in enumerate(expected_answers, start=1):
            print(f"Q{index} 标准答案: {answer}")
        return 0

    if args.answers:
        try:
            user_answers = _parse_answers(args.answers, len(questions))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    else:
        try:
            user_answers = _prompt_for_answers(questions)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2

    return _print_result(questions, user_answers, expected_answers)


if __name__ == "__main__":
    raise SystemExit(main())
