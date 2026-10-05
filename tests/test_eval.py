import asyncio

from nanogent.eval import (
    EvalTask,
    expected_tools_of,
    is_ordered_subsequence,
    run_eval_tasks,
    scripted_steps,
)


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


# ---------------------------------------------------------------------------
# 多步任务 / 禁止工具 / 有序子序列校验
# ---------------------------------------------------------------------------

def test_scripted_steps_prefers_explicit_steps():
    task = EvalTask(id="t", prompt="p", steps=[{"tool": "grep"}, {"tool": "read_file"}])
    assert [s["tool"] for s in scripted_steps(task)] == ["grep", "read_file"]


def test_scripted_steps_falls_back_to_legacy_fields():
    task = EvalTask(id="t", prompt="p", expected_tool="calculator", tool_arguments={"expression": "1+1"})
    steps = scripted_steps(task)
    assert steps == [{"tool": "calculator", "arguments": {"expression": "1+1"}}]


def test_expected_tools_supports_both_schemas():
    assert expected_tools_of(EvalTask(id="t", prompt="p", expected_tool="glob")) == ["glob"]
    multi = EvalTask(id="t", prompt="p", expected_tools=["grep", "read_file"])
    assert expected_tools_of(multi) == ["grep", "read_file"]


def test_ordered_subsequence_allows_noise_between_calls():
    assert is_ordered_subsequence(["grep", "read_file"], ["grep", "calculator", "read_file"])
    assert not is_ordered_subsequence(["read_file", "grep"], ["grep", "read_file"])
    assert is_ordered_subsequence([], [])


def test_multi_step_task_records_both_tools(tmp_path):
    task = EvalTask(
        id="multi",
        prompt="先 grep 再 read",
        steps=[{"tool": "grep", "arguments": {"pattern": "TODO", "path": "evals/fixtures"}},
               {"tool": "read_file", "arguments": {"path": "evals/fixtures/sample.py"}}],
        expected_tools=["grep", "read_file"],
        expected_substrings=["TODO"],
    )
    results = asyncio.run(run_eval_tasks([task], offline=True, out_dir=tmp_path))

    assert results[0].passed, results[0].failures
    assert results[0].called_tools == ["grep", "read_file"]


def test_forbidden_tool_fails_the_task(tmp_path):
    task = EvalTask(
        id="no_tools",
        prompt="不需要工具的问答",
        steps=[{"tool": "calculator", "arguments": {"expression": "1+1"}}],
        forbidden_tools=["calculator"],
    )
    results = asyncio.run(run_eval_tasks([task], offline=True, out_dir=tmp_path))

    assert not results[0].passed
    assert any("should not be called" in f for f in results[0].failures)


def test_wrong_tool_order_fails(tmp_path):
    task = EvalTask(
        id="wrong_order",
        prompt="顺序错了",
        steps=[{"tool": "read_file", "arguments": {"path": "evals/fixtures/sample.py"}},
               {"tool": "grep", "arguments": {"pattern": "TODO", "path": "evals/fixtures"}}],
        expected_tools=["grep", "read_file"],
    )
    results = asyncio.run(run_eval_tasks([task], offline=True, out_dir=tmp_path))

    assert not results[0].passed
    assert any("in order" in f for f in results[0].failures)
