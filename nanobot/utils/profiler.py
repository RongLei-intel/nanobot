"""Profiler instrumentation for nanobot runs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False, indent=2)


@dataclass(slots=True)
class ToolBenchmarkRecord:
    iteration: int
    name: str
    duration_ms: float
    status: str
    arguments: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    error_type: str | None = None
    error: str | None = None


@dataclass(slots=True)
class IterationBenchmarkRecord:
    iteration: int
    total_duration_ms: float = 0.0
    finish_reason: str | None = None
    stop_reason: str | None = None
    llm_input_messages: list[dict[str, Any]] = field(default_factory=list)
    llm_input_text: str = ""
    llm_output_content: str | None = None
    llm_output_reasoning_content: str | None = None
    llm_output_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_output_finish_reason: str | None = None
    llm_output_text: str = ""
    phase_durations: dict[str, float] = field(default_factory=dict)


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
    path: str | None = None
    parent_path: str | None = None
    depth: int | None = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActiveBucket:
    name: str
    start_ms: float
    category: str = "phase"
    iteration: int | None = None
    status: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    pid: int = 1
    tid: int = 1


@dataclass(slots=True)
class TreeNode:
    name: str
    path: str
    depth: int
    duration_ms: float = 0.0
    self_duration_ms: float = 0.0
    count: int = 0
    children: dict[str, "TreeNode"] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "depth": self.depth,
            "duration_ms": round(self.duration_ms, 3),
            "self_duration_ms": round(self.self_duration_ms, 3),
            "count": self.count,
            "children": [child.to_dict() for child in sorted(self.children.values(), key=lambda x: (x.path, x.name))],
        }


class ProfilerTrace:
    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = profiler_enabled()
        self.enabled = enabled
        self.session_key: str | None = None
        self.started_at: float = 0.0
        self.ended_at: float = 0.0
        self.total_duration_ms: float = 0.0
        self.iterations: list[IterationBenchmarkRecord] = []
        self.tools: list[ToolBenchmarkRecord] = []
        self.spans: list[SpanRecord] = []
        self._stack: list[ActiveBucket] = []

    def start(self, session_key: str | None = None) -> "ProfilerTrace":
        if not self.enabled:
            return self
        self.session_key = session_key
        self.started_at = time.perf_counter()
        return self

    def finish(self) -> "ProfilerTrace":
        if not self.enabled:
            return self
        while self._stack:
            self.pop(status="auto_closed")
        self.ended_at = time.perf_counter()
        self.total_duration_ms = round((self.ended_at - self.started_at) * 1000, 3)
        self._finalize_totals()
        self.write_json()
        self.write_perfetto_json()
        return self

    def now_ms(self) -> float:
        if not self.enabled:
            return 0.0
        return self._r((time.perf_counter() - self.started_at) * 1000)

    def push(self, name: str, *, category: str = "phase", iteration: int | None = None, status: str | None = None, args: dict[str, Any] | None = None, pid: int = 1, tid: int = 1) -> "ProfilerTrace":
        if not self.enabled:
            return self
        self._stack.append(ActiveBucket(name=name, start_ms=self.now_ms(), category=category, iteration=iteration, status=status, args=dict(args or {}), pid=pid, tid=tid))
        return self

    def pop(self, *, status: str | None = None, args: dict[str, Any] | None = None) -> "ProfilerTrace":
        if not self.enabled or not self._stack:
            return self
        active = self._stack.pop()
        if status is not None:
            active.status = status
        if args:
            active.args.update(args)
        end_ms = self.now_ms()
        duration_ms = self._r(max(0.0, self._r(end_ms) - self._r(active.start_ms)))
        self._apply_pop_updates(active, duration_ms)
        parent_path = self._stack_path(self._stack)
        path = self._stack_path([*self._stack, active])
        depth = len(self._stack) + 1
        self.spans.append(SpanRecord(
            name=active.name,
            category=active.category,
            start_ms=self._r(active.start_ms),
            end_ms=self._r(end_ms),
            duration_ms=duration_ms,
            pid=active.pid,
            tid=active.tid,
            iteration=active.iteration,
            status=active.status,
            path=path,
            parent_path=parent_path,
            depth=depth,
            args=dict(active.args),
        ))
        return self

    def ensure_iteration(self, iteration: int) -> IterationBenchmarkRecord:
        for item in self.iterations:
            if item.iteration == iteration:
                return item
        item = IterationBenchmarkRecord(iteration=iteration)
        self.iterations.append(item)
        self.iterations.sort(key=lambda x: x.iteration)
        return item

    def _apply_pop_updates(self, active: ActiveBucket, duration_ms: float) -> None:
        if active.category == "tool":
            self._apply_tool_pop_updates(active, duration_ms)
            return
        if active.iteration is None:
            return
        item = self.ensure_iteration(active.iteration)
        item.phase_durations[active.name] = self._r(item.phase_durations.get(active.name, 0.0) + duration_ms)
        if active.name == "llm":
            self._apply_llm_pop_updates(item, active)
        if active.name.startswith("iteration["):
            if "finish_reason" in active.args:
                item.finish_reason = active.args.get("finish_reason")
            if "stop_reason" in active.args:
                item.stop_reason = active.args.get("stop_reason")
            item.total_duration_ms = self._r(duration_ms)

    def _apply_llm_pop_updates(self, item: IterationBenchmarkRecord, active: ActiveBucket) -> None:
        messages = active.args.get("messages") or []
        response = active.args.get("response")
        input_messages = [dict(message) for message in messages]
        output_tool_calls = [tc.to_openai_tool_call() for tc in getattr(response, "tool_calls", [])]
        item.llm_input_messages = input_messages
        item.llm_input_text = _safe_json_dumps(input_messages)
        item.llm_output_content = getattr(response, "content", None)
        item.llm_output_reasoning_content = getattr(response, "reasoning_content", None)
        item.llm_output_tool_calls = output_tool_calls
        item.llm_output_finish_reason = getattr(response, "finish_reason", None)
        item.llm_output_text = _safe_json_dumps({
            "content": item.llm_output_content,
            "reasoning_content": item.llm_output_reasoning_content,
            "tool_calls": item.llm_output_tool_calls,
            "finish_reason": item.llm_output_finish_reason,
        })
        active.args = {
            "model": active.args.get("model"),
            "streaming": active.args.get("streaming"),
            "llm_input_messages": item.llm_input_messages,
            "llm_input_text": item.llm_input_text,
            "llm_output_content": item.llm_output_content,
            "llm_output_reasoning_content": item.llm_output_reasoning_content,
            "llm_output_tool_calls": item.llm_output_tool_calls,
            "llm_output_finish_reason": item.llm_output_finish_reason,
            "llm_output_text": item.llm_output_text,
        }

    def _apply_tool_pop_updates(self, active: ActiveBucket, duration_ms: float) -> None:
        iteration = active.iteration if active.iteration is not None else -1
        tool_call = active.args.get("tool_call")
        result = active.args.get("result")
        error = active.args.get("error")
        arguments = dict(getattr(tool_call, "arguments", {}) or {})
        detail = self._tool_detail(result=result, error=error)
        error_type = type(error).__name__ if error is not None else None
        error_text = str(error) if error is not None else None
        active.args = {
            "tool_name": getattr(tool_call, "name", active.name),
            "arguments": arguments,
            "detail": detail,
            "error_type": error_type,
            "error": error_text,
        }
        self.tools.append(ToolBenchmarkRecord(
            iteration=iteration,
            name=getattr(tool_call, "name", active.name),
            duration_ms=self._r(duration_ms),
            status=active.status or "ok",
            arguments=arguments,
            detail=detail,
            error_type=error_type,
            error=error_text,
        ))

    def _tool_detail(self, *, result: Any = None, error: BaseException | None = None) -> str:
        if error is not None:
            detail = str(error)
        else:
            detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            return "(empty)"
        if len(detail) > 120:
            return detail[:120] + "..."
        return detail

    def _finalize_totals(self) -> None:
        return None

    def hierarchy_tree(self) -> dict[str, Any]:
        root = TreeNode(name="root", path="", depth=0)
        nodes: dict[str, TreeNode] = {"": root}
        for span in sorted(self.spans, key=lambda x: (x.start_ms, x.end_ms, x.path or x.name)):
            path = span.path or span.name
            parent_path = span.parent_path or ""
            depth = span.depth or max(1, path.count("/") + 1)
            node = nodes.get(path)
            if node is None:
                node = TreeNode(name=span.name, path=path, depth=depth)
                nodes[path] = node
                parent = nodes.get(parent_path)
                if parent is None:
                    parent = TreeNode(name=parent_path.split("/")[-1] if parent_path else "root", path=parent_path, depth=max(0, depth - 1))
                    nodes[parent_path] = parent
                parent.children[path] = node
            node.duration_ms = self._r(node.duration_ms + span.duration_ms)
            node.count += 1
        for path, node in nodes.items():
            if path == "":
                node.duration_ms = self._r(self.total_duration_ms)
            children_total = sum(child.duration_ms for child in node.children.values())
            node.self_duration_ms = self._r(node.duration_ms - children_total)
        return root.to_dict()

    def phase_totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for span in self.spans:
            path = span.path or span.name
            totals[path] = self._r(totals.get(path, 0.0) + span.duration_ms)
        return dict(sorted(totals.items()))

    def iteration_phase_totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for item in self.iterations:
            for name, duration in item.phase_durations.items():
                totals[name] = self._r(totals.get(name, 0.0) + duration)
        return dict(sorted(totals.items()))

    @property
    def llm_total_duration_ms(self) -> float:
        return self._r(sum(item.phase_durations.get("llm", 0.0) for item in self.iterations))

    @property
    def tools_total_duration_ms(self) -> float:
        return self._r(sum(item.duration_ms for item in self.tools))

    @property
    def tools_wall_clock_total_duration_ms(self) -> float:
        return self._r(sum(item.phase_durations.get("tools_wall_clock", 0.0) for item in self.iterations))

    def summary_dict(self) -> dict[str, Any]:
        return {
            "session_key": self.session_key,
            "total_duration_ms": self._r(self.total_duration_ms),
            "iterations": len(self.iterations),
            "tool_call_count": len(self.tools),
            "span_count": len(self.spans),
            "llm_total_duration_ms": self.llm_total_duration_ms,
            "tools_total_duration_ms": self.tools_total_duration_ms,
            "tools_wall_clock_total_duration_ms": self.tools_wall_clock_total_duration_ms,
            "llm_input_char_total": sum(len(item.llm_input_text) for item in self.iterations),
            "llm_output_char_total": sum(len(item.llm_output_text) for item in self.iterations),
            "phase_totals": self.phase_totals(),
            "iteration_phase_totals": self.iteration_phase_totals(),
            "hierarchy_tree": self.hierarchy_tree(),
        }

    def summary_text(self) -> str:
        data = self.summary_dict()
        phase_totals = data["phase_totals"]
        top_items = sorted(phase_totals.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_text = ", ".join(f"{name}={value:.3f}ms" for name, value in top_items)
        return (
            "profiler summary: "
            f"run.total={data['total_duration_ms']:.3f}ms, "
            f"iterations={data['iterations']}, "
            f"llm_total={data['llm_total_duration_ms']:.3f}ms, "
            f"tools_wall_clock_total={data['tools_wall_clock_total_duration_ms']:.3f}ms, "
            f"tool_count={data['tool_call_count']}, "
            f"spans={data['span_count']}, "
            f"top_phases=[{top_text}]"
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
            "spans": [asdict(item) for item in sorted(self.spans, key=lambda x: (x.start_ms, x.tid, x.path or x.name))],
        }

    def to_perfetto_dict(self) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        thread_names: dict[int, str] = {1: "run", 10: "iterations", 20: "tools"}
        events.append({"name": "process_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "nanobot"}})
        for tid, name in thread_names.items():
            events.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": {"name": name}})
        for span in sorted(self.spans, key=lambda x: (x.start_ms, x.tid, x.path or x.name)):
            args = dict(span.args)
            if span.iteration is not None:
                args.setdefault("iteration", span.iteration)
            if span.status is not None:
                args.setdefault("status", span.status)
            if span.path is not None:
                args.setdefault("path", span.path)
            if span.parent_path is not None:
                args.setdefault("parent_path", span.parent_path)
            args.setdefault("duration_ms", span.duration_ms)
            events.append({"name": span.path or span.name, "cat": span.category, "ph": "X", "ts": self._us_from_ms(span.start_ms), "dur": self._us_from_ms(span.duration_ms), "pid": span.pid, "tid": span.tid, "args": args})
        return {"traceEvents": events, "displayTimeUnit": "ms", "metadata": {"session_key": self.session_key, "summary": self.summary_dict()}}

    def write_json(self, path: str | Path=None) -> Path:
        if path is None:
            path = profiler_trace_path()
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def write_perfetto_json(self, path: str | Path=None) -> Path:
        if path is None:
            path = profiler_perfetto_trace_path()
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_perfetto_dict(), ensure_ascii=False), encoding="utf-8")
        return target

    @staticmethod
    def _r(value: float) -> float:
        return round(value, 3)

    @staticmethod
    def _us_from_ms(value_ms: float) -> int:
        return int(round(value_ms * 1000.0))

    @staticmethod
    def _stack_path(stack: list[ActiveBucket]) -> str:
        return "/".join(item.name for item in stack)


def profiler_enabled() -> bool:
    value = os.environ.get("NANOBOT_PROFILER", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def profiler_trace_path() -> str | None:
    value = os.environ.get("NANOBOT_PROFILER_TRACE", "").strip()
    return value or None


def profiler_perfetto_trace_path() -> str | None:
    value = os.environ.get("NANOBOT_PROFILER_PERFETTO_TRACE", "").strip()
    return value or None
