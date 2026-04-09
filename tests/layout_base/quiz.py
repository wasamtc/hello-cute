#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any, Callable, List, Optional, Sequence


@dataclass(frozen=True)
class Question:
    title: str
    shape: Any
    coord: Any
    stride: Optional[Any] = None


@dataclass(frozen=True)
class Backend:
    name: str
    reason: Optional[str]
    make_layout: Callable[[Any, Optional[Any]], Any]
    crd2idx: Callable[[Any, Any], int]

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prefix_product(shape: Any, init: int = 1):
    if isinstance(shape, tuple):
        result = []
        current = init
        for value in shape:
            result.append(_prefix_product(value, current))
            current *= _product(value)
        return tuple(result)
    return init


def _product(value: Any) -> int:
    if isinstance(value, tuple):
        result = 1
        for item in value:
            result *= _product(item)
        return result
    return int(value)


def _resolved_stride(question: Question):
    if question.stride is not None:
        return question.stride
    return _prefix_product(question.shape)


def _crd2idx_python(crd: Any, shape: Any, stride: Optional[Any] = None) -> int:
    if stride is None:
        stride = _prefix_product(shape)

    if isinstance(crd, tuple):
        if not isinstance(shape, tuple):
            raise AssertionError(f"crd={crd}, shape={shape}")
        if len(crd) != len(shape) or len(crd) != len(stride):
            raise AssertionError(f"crd={crd}, shape={shape}, stride={stride}")
        return sum(
            _crd2idx_python(coord_i, shape_i, stride_i)
            for coord_i, shape_i, stride_i in zip(crd, shape, stride)
        )

    if crd is None:
        crd = 0

    if isinstance(shape, tuple):
        result = 0
        for index in range(len(shape) - 1):
            shape_i = shape[index]
            result += _crd2idx_python(
                crd % _product(shape_i),
                shape_i,
                stride[index],
            )
            crd //= _product(shape_i)
        return result + _crd2idx_python(crd, shape[-1], stride[-1])

    return int(crd) * int(stride)


def _cosize_python(shape: Any, stride: Any) -> int:
    if isinstance(shape, tuple):
        return _crd2idx_python(_last_coord(shape), shape, stride) + 1
    return _crd2idx_python(shape - 1, shape, stride) + 1


def _last_coord(shape: Any):
    if isinstance(shape, tuple):
        return tuple(_last_coord(item) for item in shape)
    return int(shape) - 1


def _random_seed() -> int:
    return random.SystemRandom().randrange(1, 10**12)


def _make_hierarchical_shape(rng: random.Random):
    return (
        rng.randint(2, 4),
        (
            rng.randint(2, 4),
            rng.randint(2, 5),
        ),
    )


def _make_hierarchical_stride(shape: Any, rng: random.Random):
    if rng.choice([True, False]):
        return None

    inner_shape = shape[1]
    inner_pad = rng.randint(0, 2)
    inner_stride = (inner_shape[1] + inner_pad, 1)
    outer_pad = rng.randint(0, 2)
    outer_stride = _cosize_python(inner_shape, inner_stride) + outer_pad
    return (outer_stride, inner_stride)


def _generate_linear_question(rng: random.Random) -> Question:
    size = rng.randint(6, 12)
    coord = rng.randint(1, size - 1)
    return Question(
        title="一维基础 layout",
        shape=size,
        coord=coord,
    )


def _generate_layoutleft_2d_question(rng: random.Random) -> Question:
    rows = rng.randint(2, 5)
    cols = rng.randint(3, 6)
    coord = (rng.randrange(rows), rng.randrange(cols))
    return Question(
        title=f"默认 LayoutLeft 的 {rows}x{cols} layout",
        shape=(rows, cols),
        coord=coord,
    )


def _generate_explicit_stride_question(rng: random.Random) -> Question:
    rows = rng.randint(2, 5)
    cols = rng.randint(3, 6)
    pad = rng.randint(0, 3)
    stride = (cols + pad, 1)
    coord = (rng.randrange(rows), rng.randrange(cols))
    if pad == 0:
        title = f"显式 stride 的 {rows}x{cols} row-major layout"
    else:
        title = f"显式 stride 的 {rows}x{cols} row-major + padding layout"
    return Question(
        title=title,
        shape=(rows, cols),
        stride=stride,
        coord=coord,
    )


def _generate_hierarchical_integer_question(rng: random.Random) -> Question:
    shape = _make_hierarchical_shape(rng)
    coord = rng.randrange(_product(shape))
    return Question(
        title="分层 layout，输入坐标是整数",
        shape=shape,
        stride=_make_hierarchical_stride(shape, rng),
        coord=coord,
    )


def _generate_hierarchical_tuple_question(rng: random.Random) -> Question:
    shape = _make_hierarchical_shape(rng)
    outer, inner = shape
    inner_rows, inner_cols = inner
    outer_coord = rng.randrange(outer)

    if rng.choice([True, False]):
        coord = (outer_coord, rng.randrange(inner_rows * inner_cols))
        title = "分层 layout，tuple 输入中内层 mode 被压平"
    else:
        coord = (
            outer_coord,
            (rng.randrange(inner_rows), rng.randrange(inner_cols)),
        )
        title = "分层 layout，tuple 输入是自然坐标"

    return Question(
        title=title,
        shape=shape,
        stride=_make_hierarchical_stride(shape, rng),
        coord=coord,
    )


def _generate_questions(rng: random.Random) -> List[Question]:
    return [
        _generate_linear_question(rng),
        _generate_layoutleft_2d_question(rng),
        _generate_explicit_stride_question(rng),
        _generate_hierarchical_integer_question(rng),
        _generate_hierarchical_tuple_question(rng),
    ]


def _format_value(value: Any) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(_format_value(item) for item in value) + ")"
    return str(value)


def _load_backend() -> Backend:
    try:
        import cutlass.cute as cute  # type: ignore

        def make_layout(shape: Any, stride: Optional[Any]):
            if stride is None:
                return cute.make_layout(shape)
            return cute.make_layout(shape, stride=stride)

        def crd2idx(coord: Any, layout: Any) -> int:
            return int(cute.crd2idx(coord, layout))

        # CuTe DSL layout algebra is JIT-oriented. Some releases can be imported
        # successfully but still cannot evaluate make_layout/crd2idx in native
        # Python context, which is exactly what this CLI quiz needs.
        probe_layout = make_layout((1, 1), (1, 1))
        _ = crd2idx((0, 0), probe_layout)

        return Backend(
            name="cutlass.cute",
            reason=None,
            make_layout=make_layout,
            crd2idx=crd2idx,
        )
    except Exception as exc:
        python_root = _repo_root() / "third_party" / "cutlass" / "python"
        if str(python_root) not in sys.path:
            sys.path.insert(0, str(python_root))

        import pycute  # type: ignore

        def make_layout(shape: Any, stride: Optional[Any]):
            return pycute.Layout(shape, stride)

        def crd2idx(coord: Any, layout: Any) -> int:
            return int(pycute.crd2idx(coord, layout.shape, layout.stride))

        return Backend(
            name="pycute",
            reason=f"{type(exc).__name__}: {exc}",
            make_layout=make_layout,
            crd2idx=crd2idx,
        )


def _compute_answers(backend: Backend, questions: Sequence[Question]) -> List[int]:
    answers: List[int] = []
    for question in questions:
        layout = backend.make_layout(question.shape, question.stride)
        answers.append(backend.crd2idx(question.coord, layout))
    return answers


def _parse_answers(raw_answers: Sequence[str], expected_count: int) -> List[int]:
    answers: List[int] = []
    for raw in raw_answers:
        for token in raw.replace(",", " ").split():
            answers.append(int(token))

    if len(answers) != expected_count:
        raise ValueError(
            f"需要 {expected_count} 个答案，实际收到 {len(answers)} 个。"
        )
    return answers


def _print_questions(questions: Sequence[Question]) -> None:
    print("请根据每个 layout 和输入坐标，写出对应的线性索引。")
    print(f"答案请按顺序输入，共 {len(questions)} 个整数，用空格分隔。")
    print()
    for index, question in enumerate(questions, start=1):
        print(f"Q{index}. {question.title}")
        print(
            f"    layout = {_format_value(question.shape)}:"
            f"{_format_value(_resolved_stride(question))}"
        )
        print(f"    input coord = {_format_value(question.coord)}")
        print()


def _prompt_for_answers(question_count: int) -> List[int]:
    raw = input(f"请输入 {question_count} 个线性索引: ").strip()
    return _parse_answers([raw], question_count)


def _print_result(
    questions: Sequence[Question],
    user_answers: Sequence[int],
    expected_answers: Sequence[int],
) -> int:
    correct = 0
    print()
    print("结果:")
    for index, (question, user_answer, expected_answer) in enumerate(
        zip(questions, user_answers, expected_answers),
        start=1,
    ):
        if user_answer == expected_answer:
            correct += 1
            print(
                f"Q{index} 正确: {user_answer}"
            )
        else:
            print(
                f"Q{index} 错误: 你的答案是 {user_answer}，正确答案是 {expected_answer}"
            )
            print(
                f"    对应题目: layout = {_format_value(question.shape)}:"
                f"{_format_value(_resolved_stride(question))}, "
                f"coord = {_format_value(question.coord)}"
            )

    print()
    print(f"总分: {correct}/{len(questions)}")
    return 0 if correct == len(questions) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="layout_base 交互测试"
    )
    parser.add_argument(
        "--answers",
        nargs="*",
        default=None,
        help="直接提供 5 个答案，便于非交互运行",
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
    if args.show_answers:
        _print_questions(questions)
        print("标准答案:", " ".join(str(answer) for answer in expected_answers))
        return 0

    _print_questions(questions)

    if args.answers:
        try:
            user_answers = _parse_answers(args.answers, len(questions))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    else:
        try:
            user_answers = _prompt_for_answers(len(questions))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2

    return _print_result(questions, user_answers, expected_answers)


if __name__ == "__main__":
    raise SystemExit(main())
