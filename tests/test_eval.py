import asyncio

from nanogent.eval import EvalTask, run_eval_tasks


def test_offline_eval_generates_report(tmp_path) -> None:
    tasks = [
        EvalTask(
            id="calculator_basic",
            prompt="Calculate 2 + 3.",
            expected_tool="calculator",
            tool_arguments={"expression": "2 + 3"},
            expected_substrings=["5"],
        )
    ]

    results = asyncio.run(run_eval_tasks(tasks, offline=True, out_dir=tmp_path))

    assert len(results) == 1
    assert results[0].passed
    assert results[0].called_tools == ["calculator"]
    assert (tmp_path / "calculator_basic.trace.jsonl").exists()
    assert (tmp_path / "eval_results.json").exists()
    assert (tmp_path / "eval_report.md").exists()
