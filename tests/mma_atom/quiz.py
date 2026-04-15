#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any, List, Sequence


@dataclass(frozen=True)
class LayoutSpec:
    shape: Any
    stride: Any


@dataclass(frozen=True)
class MmaOperationSpec:
    operation_expr: str
    operation_name: str
    header_path: str
    clayout: LayoutSpec


@dataclass(frozen=True)
class Question:
    title: str
    operation: MmaOperationSpec


MMA_OPERATION_POOL: Sequence[MmaOperationSpec] = (
    MmaOperationSpec(
        operation_expr="SM70_8x8x4_F16F16F16F16_TN",
        operation_name="SM70_8x8x4_F16F16F16F16_TN",
        header_path="third_party/cutlass/include/cute/atom/mma_traits_sm70.hpp",
        clayout=LayoutSpec(
            shape=(8, 8),
            stride=(1, 8),
        ),
    ),
    MmaOperationSpec(
        operation_expr="SM70_8x8x4_F32F16F16F32_NT",
        operation_name="SM70_8x8x4_F32F16F16F32_NT",
        header_path="third_party/cutlass/include/cute/atom/mma_traits_sm70.hpp",
        clayout=LayoutSpec(
            shape=((2, 2, 2), (2, 2, 2)),
            stride=((1, 16, 4), (8, 2, 32)),
        ),
    ),
    MmaOperationSpec(
        operation_expr="SM75_8x8x16_S32S8S8S32_TN",
        operation_name="SM75_8x8x16_S32S8S8S32_TN",
        header_path="third_party/cutlass/include/cute/atom/mma_traits_sm75.hpp",
        clayout=LayoutSpec(
            shape=((4, 8), 2),
            stride=((16, 1), 8),
        ),
    ),
    MmaOperationSpec(
        operation_expr="SM75_16x8x8_F32F16F16F32_TN",
        operation_name="SM75_16x8x8_F32F16F16F32_TN",
        header_path="third_party/cutlass/include/cute/atom/mma_traits_sm75.hpp",
        clayout=LayoutSpec(
            shape=((4, 8), (2, 2)),
            stride=((32, 1), (16, 8)),
        ),
    ),
    MmaOperationSpec(
        operation_expr="SM90_64x8x16_F32F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>",
        operation_name="SM90_64x8x16_F32F16F16_SS",
        header_path="third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp",
        clayout=LayoutSpec(
            shape=((4, 8, 4), (2, 2, 1)),
            stride=((128, 1, 16), (64, 8, 512)),
        ),
    ),
    MmaOperationSpec(
        operation_expr="SM90_64x16x16_F32F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>",
        operation_name="SM90_64x16x16_F32F16F16_SS",
        header_path="third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp",
        clayout=LayoutSpec(
            shape=((4, 8, 4), (2, 2, 2)),
            stride=((128, 1, 16), (64, 8, 512)),
        ),
    ),
    MmaOperationSpec(
        operation_expr="SM90_64x32x16_F32F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>",
        operation_name="SM90_64x32x16_F32F16F16_SS",
        header_path="third_party/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp",
        clayout=LayoutSpec(
            shape=((4, 8, 4), (2, 2, 4)),
            stride=((128, 1, 16), (64, 8, 512)),
        ),
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _verify_operation_pool() -> None:
    repo_root = _repo_root()
    for op in MMA_OPERATION_POOL:
        header_text = (repo_root / op.header_path).read_text()
        if op.operation_name not in header_text:
            raise ValueError(
                f"operation {op.operation_name} 不存在于 {op.header_path}"
            )


def _generate_questions(rng: random.Random) -> List[Question]:
    sample_size = min(3, len(MMA_OPERATION_POOL))
    operations = rng.sample(list(MMA_OPERATION_POOL), sample_size)
    return [
        Question(
            title="给定真实的 MMA operation，请写出对应的 CLayout",
            operation=operation,
        )
        for operation in operations
    ]


def _compute_answers(questions: Sequence[Question]) -> List[str]:
    return [_format_layout(question.operation.clayout) for question in questions]


def _print_questions(questions: Sequence[Question]) -> None:
    print("请写出下面每个 MMA operation 对应的 `MMA_Traits<Operation>::CLayout`。")
    print("答案请按 `shape:stride` 的完整格式输入，空格不敏感。")
    print("交互模式下每题输入一行。")
    print()
    for index, question in enumerate(questions, start=1):
        print(f"Q{index}. {question.title}")
        print(f"    Operation = {question.operation.operation_expr}")
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
        answers.append(input(f"请输入 Q{index} 的 CLayout: ").strip())
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
            print(f"    对应题目: {question.operation.operation_expr}")

    print()
    print(f"总分: {correct}/{len(questions)}")
    return 0 if correct == len(questions) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="mma_atom 交互测试"
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

    _verify_operation_pool()

    seed = args.seed if args.seed is not None else _random_seed()
    rng = random.Random(seed)
    questions = _generate_questions(rng)

    print("当前测试对象: 真实 MMA operation")
    print(f"题目 seed: {seed}")
    print()

    expected_answers = _compute_answers(questions)
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
