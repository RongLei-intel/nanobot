"""Benchmark instrumentation for nanobot agent runs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ToolBenchmarkRecord:
    iteration: int
    name: str
    duration_ms: float
    status: str


@dataclass(slots=True)
class IterationBenchmarkRecord:
    iteration: int
    llm_duration_ms: float = 0.0
    tool_count: int = 0
    tools_duration_ms: float = 0.0
    tools_wall_clock_duration_ms: float = 0.0
    hook_before_iteration_duration_ms: float = 0.0
    hook_before_execute_tools_duration_ms: float = 0.0
    hook_after_iteration_duration_ms: float = 0.0
    message_build_assistant_duration_ms: float = 0.0
    message_build_tool_results_duration_ms: float = 0.0
    finalize_content_duration_ms: float = 0.0
    framework_overhead_duration_ms: float = 0.0
    accounted_total_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    finish_reason: str | None = None
    stop_reason: str | None = None


@dataclass(slots=True)
class SpanRecord:
    name: str
    category: str
    start_ms: float
    end_ms: float
    duration_ms: float
    pid: int = 1
    tid: int = 1
    iteration: int | None = None
    status: str | None = None
    args: dict[str, Any] = field(default_factory=dict)


class BenchmarkSpan:
    __slots__ = ("_trace", "_name", "_category", "_pid", "_tid", "_iteration", "_status", "_args", "_start")

    def __init__(
        self,
        trace: BenchmarkTrace,
        *,
        name: str,
        category: str,
        pid: int = 1,
        tid: int = 1,
        iteration: int | None = None,
        status: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        self._trace = trace
        self._name = name
        self._category = category
        self._pid = pid
        self._tid = tid
        self._iteration = iteration
        self._status = status
        self._args = dict(args or {})
        self._start = 0.0

    def __enter__(self) -> BenchmarkSpan:
        self._start = self._trace.now_ms()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None and self._status is None:
            self._status = "error"
            self._args.setdefault("error_type", type(exc).__name__)
            self._args.setdefault("error", str(exc))
        self._trace.add_span(
            name=self._name,
            category=self._category,
            start_ms=self._start,
            end_ms=self._trace.now_ms(),
            pid=self._pid,
            tid=self._tid,
            iteration=self._iteration,
            status=self._status,
            args=self._args,
        )


@dataclass(slots=True)
class BenchmarkTrace:
    enabled: bool = False
    session_key: str | None = None
    started_at: float = 0.0
    ended_at: float = 0.0
    total_duration_ms: float = 0.0
    connect_mcp_duration_ms: float = 0.0
    session_load_duration_ms: float = 0.0
    command_dispatch_duration_ms: float = 0.0
    memory_consolidation_before_duration_ms: float = 0.0
    tool_context_setup_duration_ms: float = 0.0
    turn_setup_duration_ms: float = 0.0
    context_build_duration_ms: float = 0.0
    agent_loop_duration_ms: float = 0.0
    save_turn_duration_ms: float = 0.0
    session_save_duration_ms: float = 0.0
    background_schedule_duration_ms: float = 0.0
    response_build_duration_ms: float = 0.0
    framework_overhead_duration_ms: float = 0.0
    accounted_total_duration_ms: float = 0.0
    iterations: list[IterationBenchmarkRecord] = field(default_factory=list)
    tools: list[ToolBenchmarkRecord] = field(default_factory=list)
    spans: list[SpanRecord] = field(default_factory=list)

    def start(self, session_key: str | None = None) -> None:
        self.enabled = True
        self.session_key = session_key
        self.started_at = time.perf_counter()

    def finish(self) -> None:
        if not self.enabled:
            return
        self.ended_at = time.perf_counter()
        self.total_duration_ms = round((self.ended_at - self.started_at) * 1000, 3)
        self._finalize_totals()

    def now_ms(self) -> float:
        if not self.enabled:
            return 0.0
        return self._r((time.perf_counter() - self.started_at) * 1000)

    def span(
        self,
        *,
        name: str,
        category: str,
        pid: int = 1,
        tid: int = 1,
        iteration: int | None = None,
        status: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> BenchmarkSpan:
        return BenchmarkSpan(
            self,
            name=name,
            category=category,
            pid=pid,
            tid=tid,
            iteration=iteration,
            status=status,
            args=args,
        )

    def add_span(
        self,
        *,
        name: str,
        category: str,
        start_ms: float,
        end_ms: float,
        pid: int = 1,
        tid: int = 1,
        iteration: int | None = None,
        status: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        start_ms = self._r(start_ms)
        end_ms = self._r(end_ms)
        duration_ms = self._r(max(0.0, end_ms - start_ms))
        self.spans.append(SpanRecord(
            name=name,
            category=category,
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=duration_ms,
            pid=pid,
            tid=tid,
            iteration=iteration,
            status=status,
            args=dict(args or {}),
        ))

    def ensure_iteration(self, iteration: int) -> IterationBenchmarkRecord:
        for item in self.iterations:
            if item.iteration == iteration:
                return item
        item = IterationBenchmarkRecord(iteration=iteration)
        self.iterations.append(item)
        self.iterations.sort(key=lambda x: x.iteration)
        return item

    @staticmethod
    def _r(value: float) -> float:
        return round(value, 3)

    @staticmethod
    def _us_from_ms(value_ms: float) -> int:
        return int(round(value_ms * 1000.0))

    def add_top_level_duration(self, field_name: str, duration_ms: float) -> None:
        current = getattr(self, field_name)
        setattr(self, field_name, self._r(current + duration_ms))

    def add_iteration_duration(self, iteration: int, field_name: str, duration_ms: float) -> None:
        item = self.ensure_iteration(iteration)
        current = getattr(item, field_name)
        setattr(item, field_name, self._r(current + duration_ms))

    def add_llm_duration(self, iteration: int, duration_ms: float) -> None:
        item = self.ensure_iteration(iteration)
        item.llm_duration_ms = self._r(duration_ms)

    def add_tool_record(self, iteration: int, name: str, duration_ms: float, status: str) -> None:
        self.tools.append(ToolBenchmarkRecord(
            iteration=iteration,
            name=name,
            duration_ms=self._r(duration_ms),
            status=status,
        ))
        item = self.ensure_iteration(iteration)
        item.tool_count += 1
        item.tools_duration_ms = self._r(item.tools_duration_ms + duration_ms)

    def set_tools_wall_clock_duration(self, iteration: int, duration_ms: float) -> None:
        item = self.ensure_iteration(iteration)
        item.tools_wall_clock_duration_ms = self._r(duration_ms)

    def finalize_iteration(
        self,
        iteration: int,
        *,
        finish_reason: str | None = None,
        stop_reason: str | None = None,
        total_duration_ms: float | None = None,
    ) -> None:
        item = self.ensure_iteration(iteration)
        if finish_reason is not None:
            item.finish_reason = finish_reason
        if stop_reason is not None:
            item.stop_reason = stop_reason
        if total_duration_ms is not None:
            item.total_duration_ms = self._r(total_duration_ms)
        self._finalize_iteration(item)

    def _finalize_iteration(self, item: IterationBenchmarkRecord) -> None:
        accounted = (
            item.llm_duration_ms
            + item.tools_wall_clock_duration_ms
            + item.hook_before_iteration_duration_ms
            + item.hook_before_execute_tools_duration_ms
            + item.hook_after_iteration_duration_ms
            + item.message_build_assistant_duration_ms
            + item.message_build_tool_results_duration_ms
            + item.finalize_content_duration_ms
        )
        item.accounted_total_duration_ms = self._r(accounted)
        item.framework_overhead_duration_ms = self._r(item.total_duration_ms - item.accounted_total_duration_ms)
        item.accounted_total_duration_ms = self._r(
            item.accounted_total_duration_ms + item.framework_overhead_duration_ms
        )

    def _finalize_totals(self) -> None:
        for item in self.iterations:
            self._finalize_iteration(item)
        accounted = (
            self.connect_mcp_duration_ms
            + self.session_load_duration_ms
            + self.command_dispatch_duration_ms
            + self.memory_consolidation_before_duration_ms
            + self.tool_context_setup_duration_ms
            + self.turn_setup_duration_ms
            + self.context_build_duration_ms
            + self.agent_loop_duration_ms
            + self.save_turn_duration_ms
            + self.session_save_duration_ms
            + self.background_schedule_duration_ms
            + self.response_build_duration_ms
        )
        self.accounted_total_duration_ms = self._r(accounted)
        self.framework_overhead_duration_ms = self._r(self.total_duration_ms - self.accounted_total_duration_ms)
        self.accounted_total_duration_ms = self._r(
            self.accounted_total_duration_ms + self.framework_overhead_duration_ms
        )

    @property
    def llm_total_duration_ms(self) -> float:
        return self._r(sum(item.llm_duration_ms for item in self.iterations))

    @property
    def tools_total_duration_ms(self) -> float:
        return self._r(sum(item.duration_ms for item in self.tools))

    @property
    def tools_wall_clock_total_duration_ms(self) -> float:
        return self._r(sum(item.tools_wall_clock_duration_ms for item in self.iterations))

    @property
    def iteration_framework_overhead_total_duration_ms(self) -> float:
        return self._r(sum(item.framework_overhead_duration_ms for item in self.iterations))

    def summary_dict(self) -> dict[str, Any]:
        return {
            "session_key": self.session_key,
            "total_duration_ms": self._r(self.total_duration_ms),
            "accounted_total_duration_ms": self._r(self.accounted_total_duration_ms),
            "framework_overhead_duration_ms": self._r(self.framework_overhead_duration_ms),
            "connect_mcp_duration_ms": self.connect_mcp_duration_ms,
            "session_load_duration_ms": self.session_load_duration_ms,
            "command_dispatch_duration_ms": self.command_dispatch_duration_ms,
            "memory_consolidation_before_duration_ms": self.memory_consolidation_before_duration_ms,
            "tool_context_setup_duration_ms": self.tool_context_setup_duration_ms,
            "turn_setup_duration_ms": self.turn_setup_duration_ms,
            "context_build_duration_ms": self.context_build_duration_ms,
            "agent_loop_duration_ms": self.agent_loop_duration_ms,
            "save_turn_duration_ms": self.save_turn_duration_ms,
            "session_save_duration_ms": self.session_save_duration_ms,
            "background_schedule_duration_ms": self.background_schedule_duration_ms,
            "response_build_duration_ms": self.response_build_duration_ms,
            "iterations": len(self.iterations),
            "llm_total_duration_ms": self.llm_total_duration_ms,
            "tools_total_duration_ms": self.tools_total_duration_ms,
            "tools_wall_clock_total_duration_ms": self.tools_wall_clock_total_duration_ms,
            "iteration_framework_overhead_total_duration_ms": self.iteration_framework_overhead_total_duration_ms,
            "tool_call_count": len(self.tools),
            "span_count": len(self.spans),
        }

    def summary_text(self) -> str:
        data = self.summary_dict()
        return (
            "benchmark summary: "
            f"total={data['total_duration_ms']:.3f}ms, "
            f"accounted={data['accounted_total_duration_ms']:.3f}ms, "
            f"top_overhead={data['framework_overhead_duration_ms']:.3f}ms, "
            f"context_build={data['context_build_duration_ms']:.3f}ms, "
            f"agent_loop={data['agent_loop_duration_ms']:.3f}ms, "
            f"iterations={data['iterations']}, "
            f"llm={data['llm_total_duration_ms']:.3f}ms, "
            f"tools_sum={data['tools_total_duration_ms']:.3f}ms, "
            f"tools_wall={data['tools_wall_clock_total_duration_ms']:.3f}ms, "
            f"tool_calls={data['tool_call_count']}, "
            f"spans={data['span_count']}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "session_key": self.session_key,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_duration_ms": self._r(self.total_duration_ms),
            "summary": self.summary_dict(),
            "iterations": [asdict(item) for item in self.iterations],
            "tools": [asdict(item) for item in self.tools],
            "spans": [asdict(item) for item in sorted(self.spans, key=lambda x: (x.start_ms, x.tid, x.name))],
        }

    def to_perfetto_dict(self) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        thread_names: dict[int, str] = {
            1: "run",
            2: "top-level",
            10: "iterations",
            20: "tools",
        }

        events.append({"name": "process_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "nanobot"}})
        for tid, name in thread_names.items():
            events.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": {"name": name}})

        for span in sorted(self.spans, key=lambda x: (x.start_ms, x.tid, x.name)):
            args = dict(span.args)
            if span.iteration is not None:
                args.setdefault("iteration", span.iteration)
            if span.status is not None:
                args.setdefault("status", span.status)
            args.setdefault("duration_ms", span.duration_ms)
            events.append({
                "name": span.name,
                "cat": span.category,
                "ph": "X",
                "ts": self._us_from_ms(span.start_ms),
                "dur": self._us_from_ms(span.duration_ms),
                "pid": span.pid,
                "tid": span.tid,
                "args": args,
            })

        return {
            "traceEvents": events,
            "displayTimeUnit": "ms",
            "metadata": {
                "session_key": self.session_key,
                "summary": self.summary_dict(),
            },
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def write_perfetto_json(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_perfetto_dict(), ensure_ascii=False), encoding="utf-8")
        return target


def benchmark_enabled() -> bool:
    value = os.environ.get("NANOBOT_BENCHMARK", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def benchmark_trace_path() -> str | None:
    value = os.environ.get("NANOBOT_BENCHMARK_TRACE", "").strip()
    return value or None


def benchmark_perfetto_trace_path() -> str | None:
    value = os.environ.get("NANOBOT_BENCHMARK_PERFETTO_TRACE", "").strip()
    return value or None
