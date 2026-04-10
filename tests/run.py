#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


TEST_RUNNERS = {
    "layout_base": "layout_base.quiz",
    "layout_algebra": "layout_algebra.quiz",
}


def _load_runner(test_name: str):
    tests_dir = Path(__file__).resolve().parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    module_name = TEST_RUNNERS.get(test_name)
    if module_name is None:
        available = ", ".join(sorted(TEST_RUNNERS))
        raise SystemExit(
            f"未知测试名称: {test_name}\n可用测试: {available}"
        )

    module = __import__(module_name, fromlist=["main"])
    return module.main


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="运行 hello-cute 交互测试"
    )
    parser.add_argument(
        "test_name",
        help="测试名称，例如: layout_base",
    )
    args, remaining = parser.parse_known_args(argv)
    runner = _load_runner(args.test_name)
    return runner(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
