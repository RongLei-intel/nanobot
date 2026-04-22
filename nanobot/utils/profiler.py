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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_openai_tool_call"):
        try:
            return _json_safe(value.to_openai_tool_call())
        except Exception:
            return str(value)
    if hasattr(value, "arguments") and hasattr(value, "name"):
        return {
            "name": getattr(value, "name", None),
            "arguments": _json_safe(getattr(value, "arguments", None)),
            "id": getattr(value, "id", None),
        }
    return str(value)


@dataclass(slots=True)
class ToolBenchmarkRecord:
    sibling_index: int
    name: str
    duration_ms: float
    status: str
    arguments: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    error_type: str | None = None
    error: str | None = None


@dataclass(slots=True)
class SpanRecord:
    name: str
    category: str
    start_ms: float
    end_ms: float
    duration_ms: float
    pid: int = 1
    tid: int = 1
    sibling_index: int = 0
    status: str | None = None
    path: str | None = None
    display_path: str | None = None
    parent_path: str | None = None
    display_parent_path: str | None = None
    depth: int | None = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActiveBucket:
    name: str
    start_ms: float
    category: str = "mainloop"
    sibling_index: int = 0
    status: str | None = None
    path: str | None = None
    parent_path: str | None = None
    display_path: str | None = None
    display_parent_path: str | None = None
    child_counts: dict[str, int] = field(default_factory=dict)
    args: dict[str, Any] = field(default_factory=dict)


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
    """Collects nested profiling spans for a single nanobot run.

    Typical usage is:

        profiler = ProfilerTrace().start(session_key)
        profiler.push("run")
        ...
        profiler.pop()
        profiler.finish()

    Spans are recorded as a stack. Each ``push()`` opens a span and each
    matching ``pop()`` closes the most recent active span. When ``finish()`` is
    called, any remaining open spans are auto-closed and the trace is exported
    to both the JSON summary format and the Perfetto trace format.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = profiler_enabled()
        self.enabled = enabled
        self.session_key: str | None = None
        self.started_at: float = 0.0
        self.ended_at: float = 0.0
        self.total_duration_ms: float = 0.0
        self.tools: list[ToolBenchmarkRecord] = []
        self.spans: list[SpanRecord] = []
        self._stack: list[ActiveBucket] = []
        self._root_child_counts: dict[str, int] = {}

    def start(self, session_key: str | None = None) -> "ProfilerTrace":
        """Start a new profiling session.

        Args:
            session_key: Optional stable identifier for the current run. This is
                included in exported trace metadata and summaries.

        Returns:
            The profiler instance itself so callers can chain setup calls.
        """
        if not self.enabled:
            return self
        self.session_key = session_key
        self.started_at = time.perf_counter()
        return self

    def finish(self) -> "ProfilerTrace":
        """Finalize the profiling session and write trace outputs.

        Any spans still left on the stack are closed automatically with the
        status ``"auto_closed"``. The final summary JSON and Perfetto trace are
        then written to the configured profiler output paths.

        Returns:
            The profiler instance itself.
        """
        if not self.enabled:
            return self
        while self._stack:
            self.pop(status="auto_closed")
        self.ended_at = time.perf_counter()
        self.total_duration_ms = self._r((self.ended_at - self.started_at) * 1000)
        self.write_json()
        self.write_perfetto_json()
        return self

    def now_ms(self) -> float:
        if not self.enabled:
            return 0.0
        return self._r((time.perf_counter() - self.started_at) * 1000)

    def push(self, name: str, *, category: str = "mainloop", status: str | None = None) -> "ProfilerTrace":
        """Open a new profiling span.

        The new span becomes the current active span and is nested under the
        current stack top, if any. Repeated sibling spans with the same name are
        assigned a local ``sibling_index`` for display and export purposes.

        ``push()`` is intentionally minimal: it only opens the span and records
        its structural metadata. End-of-span payloads should be attached in
        ``pop()``.

        Args:
            name: Logical span name, such as ``"agent_loop"`` or ``"llm_call"``.
            category: High-level lane used for reporting and Perfetto export.
                Expected values are typically ``"mainloop"``, ``"model"``, or
                ``"tool"``.
            status: Optional initial status attached to the span.

        Returns:
            The profiler instance itself.
        """
        if not self.enabled:
            return self
        parent = self._stack[-1] if self._stack else None
        parent_path = parent.path or "" if parent else ""
        display_parent_path = parent.display_path or "" if parent else ""
        sibling_index = self._next_sibling_index(parent=parent, name=name)
        display_path = self._join_path(display_parent_path, f"{name}[{sibling_index}]")
        self._stack.append(ActiveBucket(
            name=name,
            start_ms=self.now_ms(),
            category=category,
            sibling_index=sibling_index,
            status=status,
            path=self._join_path(parent_path, name),
            parent_path=parent_path,
            display_path=display_path,
            display_parent_path=display_parent_path,
            args={},
        ))
        return self

    def pop(self, *, status: str | None = None, args: dict[str, Any] | None = None) -> "ProfilerTrace":
        """Close the most recently opened active span.

        This method updates the active span with any final status or metadata,
        computes its duration, normalizes category-specific payloads, and stores
        the completed span in the exported trace list.

        Args:
            status: Optional final status for the span.
            args: Optional metadata merged into the span before it is finalized.
                This is the preferred place to attach end-of-span data such as
                model responses, tool results, or errors.

        Returns:
            The profiler instance itself. If profiling is disabled or the stack
            is empty, this is a no-op.
        """
        if not self.enabled or not self._stack:
            return self
        active = self._stack.pop()
        if status is not None:
            active.status = status
        if args:
            active.args.update(args)
        end_ms = self.now_ms()
        duration_ms = self._r(max(0.0, end_ms - active.start_ms))
        self._apply_pop_updates(active, duration_ms)
        pid, tid = self._lane_for_category(active.category)
        self.spans.append(SpanRecord(
            name=(active.display_path or active.name).rsplit("/", 1)[-1],
            category=active.category,
            start_ms=self._r(active.start_ms),
            end_ms=self._r(end_ms),
            duration_ms=duration_ms,
            pid=pid,
            tid=tid,
            sibling_index=active.sibling_index,
            status=active.status,
            path=active.path,
            display_path=active.display_path,
            parent_path=active.parent_path or "",
            display_parent_path=active.display_parent_path,
            depth=len(self._stack) + 1,
            args=dict(active.args),
        ))
        return self

    def _next_sibling_index(self, *, parent: ActiveBucket | None, name: str) -> int:
        counts = parent.child_counts if parent is not None else self._root_child_counts
        index = counts.get(name, 0)
        counts[name] = index + 1
        return index

    @staticmethod
    def _join_path(parent: str, name: str) -> str:
        return f"{parent}/{name}" if parent else name

    def _apply_pop_updates(self, active: ActiveBucket, duration_ms: float) -> None:
        if active.category == "model":
            if "messages" in active.args or "response" in active.args:
                self._apply_llm_pop_updates(active, duration_ms)
        elif active.category == "tool":
            if "tool_call" in active.args or "result" in active.args or "error" in active.args:
                self._apply_tool_pop_updates(active, duration_ms)
        active.args = _json_safe(active.args)

    def _apply_llm_pop_updates(self, active: ActiveBucket, duration_ms: float) -> None:
        del duration_ms
        messages = active.args.get("messages") or []
        response = active.args.get("response")
        input_messages = [dict(message) for message in messages]
        output_tool_calls = [tc.to_openai_tool_call() for tc in getattr(response, "tool_calls", [])]
        llm_input_text = _safe_json_dumps(input_messages)
        llm_output_content = getattr(response, "content", None)
        llm_output_reasoning_content = getattr(response, "reasoning_content", None)
        llm_output_finish_reason = getattr(response, "finish_reason", None)
        llm_output_text = _safe_json_dumps({
            "content": llm_output_content,
            "reasoning_content": llm_output_reasoning_content,
            "tool_calls": output_tool_calls,
            "finish_reason": llm_output_finish_reason,
        })
        active.args = {
            "model": active.args.get("model"),
            "llm_input_messages": input_messages,
            "llm_input_text": llm_input_text,
            "llm_output_content": llm_output_content,
            "llm_output_reasoning_content": llm_output_reasoning_content,
            "llm_output_tool_calls": output_tool_calls,
            "llm_output_finish_reason": llm_output_finish_reason,
            "llm_output_text": llm_output_text,
        }

    def _apply_tool_pop_updates(self, active: ActiveBucket, duration_ms: float) -> None:
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
            sibling_index=active.sibling_index,
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

    def hierarchy_tree(self) -> dict[str, Any]:
        root = TreeNode(name="root", path="", depth=0)
        nodes: dict[str, TreeNode] = {"": root}
        for span in sorted(self.spans, key=lambda x: (x.start_ms, x.end_ms, x.display_path or x.name)):
            path = span.display_path or span.name
            parent_path = span.display_parent_path or ""
            depth = span.depth or max(1, (span.display_path or span.name).count("/") + 1)
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
            path = span.display_path or span.name
            totals[path] = self._r(totals.get(path, 0.0) + span.duration_ms)
        return dict(sorted(totals.items()))

    @property
    def llm_total_duration_ms(self) -> float:
        return self._r(sum(span.duration_ms for span in self.spans if span.category == "model"))

    @property
    def tools_total_duration_ms(self) -> float:
        return self._r(sum(
            span.duration_ms
            for span in self.spans
            if span.category == "tool" and not self._is_tool_batch_span(span)
        ))

    @property
    def tools_wall_clock_total_duration_ms(self) -> float:
        return self._r(sum(
            span.duration_ms
            for span in self.spans
            if self._is_tool_batch_span(span)
        ))

    def summary_dict(self) -> dict[str, Any]:
        llm_input_char_total = 0
        llm_output_char_total = 0
        for span in self.spans:
            if span.category != "model":
                continue
            llm_input_char_total += len(str(span.args.get("llm_input_text") or ""))
            llm_output_char_total += len(str(span.args.get("llm_output_text") or ""))
        return {
            "session_key": self.session_key,
            "total_duration_ms": self._r(self.total_duration_ms),
            "tool_call_count": len(self.tools),
            "span_count": len(self.spans),
            "llm_total_duration_ms": self.llm_total_duration_ms,
            "tools_total_duration_ms": self.tools_total_duration_ms,
            "tools_wall_clock_total_duration_ms": self.tools_wall_clock_total_duration_ms,
            "llm_input_char_total": llm_input_char_total,
            "llm_output_char_total": llm_output_char_total,
            "phase_totals": self.phase_totals(),
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
            "tools": [asdict(item) for item in self.tools],
            "spans": [asdict(item) for item in sorted(self.spans, key=lambda x: (x.start_ms, x.tid, x.display_path or x.name))],
        }

    def to_perfetto_dict(self) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        lane_names: dict[int, str] = {
            1: "mainloop",
            2: "model",
            3: "tool",
        }
        events.append({"name": "process_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "nanobot"}})
        for tid, name in lane_names.items():
            events.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": {"name": name}})
        for span in sorted(self.spans, key=lambda x: (x.start_ms, x.tid, x.display_path or x.name)):
            args = dict(span.args)
            args.setdefault("sibling_index", span.sibling_index)
            if span.status is not None:
                args.setdefault("status", span.status)
            if span.path is not None:
                args.setdefault("path", span.path)
            if span.display_path is not None:
                args.setdefault("display_path", span.display_path)
            if span.parent_path is not None:
                args.setdefault("parent_path", span.parent_path)
            if span.display_parent_path is not None:
                args.setdefault("display_parent_path", span.display_parent_path)
            args.setdefault("duration_ms", span.duration_ms)
            events.append({"name": span.display_path or span.name, "cat": span.category, "ph": "X", "ts": self._us_from_ms(span.start_ms), "dur": self._us_from_ms(span.duration_ms), "pid": span.pid, "tid": span.tid, "args": args})
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
    def _is_tool_batch_span(span: SpanRecord) -> bool:
        path = span.path or ""
        return span.category == "tool" and path.split("/")[-1] == "execute_tools"

    @staticmethod
    def _lane_for_category(category: str) -> tuple[int, int]:
        lane_map = {
            "mainloop": (1, 1),
            "model": (1, 2),
            "tool": (1, 3),
        }
        return lane_map.get(category, (1, 1))


def profiler_enabled() -> bool:
    value = os.environ.get("NANOBOT_PROFILER", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def profiler_trace_path() -> str | None:
    value = os.environ.get("NANOBOT_PROFILER_TRACE", "").strip()
    return value or None


def profiler_perfetto_trace_path() -> str | None:
    value = os.environ.get("NANOBOT_PROFILER_PERFETTO_TRACE", "").strip()
    return value or None
