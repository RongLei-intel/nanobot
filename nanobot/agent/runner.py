"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from nanobot.agent.benchmark import BenchmarkTrace
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider, ToolCallRequest
from nanobot.utils.helpers import build_assistant_message

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    benchmark: BenchmarkTrace | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    benchmark: BenchmarkTrace | None = None


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []

        for iteration in range(spec.max_iterations):
            iteration_started = time.perf_counter()
            iteration_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            context = AgentHookContext(iteration=iteration, messages=messages, benchmark=spec.benchmark)

            started = time.perf_counter()
            hook_before_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            await hook.before_iteration(context)
            hook_before_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            if spec.benchmark is not None:
                spec.benchmark.add_iteration_duration(
                    iteration, "hook_before_iteration_duration_ms", (time.perf_counter() - started) * 1000,
                )
                spec.benchmark.add_span(
                    name="hook_before_iteration",
                    category="iteration_phase",
                    start_ms=hook_before_start_ms,
                    end_ms=hook_before_end_ms,
                    tid=10,
                    iteration=iteration,
                )

            kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": spec.tools.get_definitions(),
                "model": spec.model,
            }
            if spec.temperature is not None:
                kwargs["temperature"] = spec.temperature
            if spec.max_tokens is not None:
                kwargs["max_tokens"] = spec.max_tokens
            if spec.reasoning_effort is not None:
                kwargs["reasoning_effort"] = spec.reasoning_effort

            llm_started = time.perf_counter()
            llm_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            if hook.wants_streaming():
                async def _stream(delta: str) -> None:
                    await hook.on_stream(context, delta)

                response = await self.provider.chat_stream_with_retry(
                    **kwargs,
                    on_content_delta=_stream,
                )
            else:
                response = await self.provider.chat_with_retry(**kwargs)
            llm_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            llm_duration_ms = (time.perf_counter() - llm_started) * 1000
            if spec.benchmark is not None:
                output_tool_calls = [tc.to_openai_tool_call() for tc in response.tool_calls]
                spec.benchmark.add_llm_duration(iteration, llm_duration_ms)
                spec.benchmark.set_llm_exchange(
                    iteration,
                    input_messages=[dict(message) for message in messages],
                    output_content=response.content,
                    output_reasoning_content=response.reasoning_content,
                    output_tool_calls=output_tool_calls,
                    output_finish_reason=response.finish_reason,
                )
                spec.benchmark.add_span(
                    name="llm",
                    category="iteration_phase",
                    start_ms=llm_start_ms,
                    end_ms=llm_end_ms,
                    tid=10,
                    iteration=iteration,
                    args={
                        "model": spec.model,
                        "streaming": hook.wants_streaming(),
                        "llm_input_text": spec.benchmark.ensure_iteration(iteration).llm_input_text,
                        "llm_output_text": spec.benchmark.ensure_iteration(iteration).llm_output_text,
                    },
                )

            raw_usage = response.usage or {}
            usage = {
                "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
            }
            context.response = response
            context.usage = usage
            context.tool_calls = list(response.tool_calls)

            if response.has_tool_calls:
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)

                started = time.perf_counter()
                assistant_build_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                messages.append(build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                ))
                assistant_build_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                if spec.benchmark is not None:
                    spec.benchmark.add_iteration_duration(
                        iteration,
                        "message_build_assistant_duration_ms",
                        (time.perf_counter() - started) * 1000,
                    )
                    spec.benchmark.add_span(
                        name="message_build_assistant",
                        category="iteration_phase",
                        start_ms=assistant_build_start_ms,
                        end_ms=assistant_build_end_ms,
                        tid=10,
                        iteration=iteration,
                        args={"has_tool_calls": True},
                    )
                tools_used.extend(tc.name for tc in response.tool_calls)

                started = time.perf_counter()
                before_tools_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                await hook.before_execute_tools(context)
                before_tools_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                if spec.benchmark is not None:
                    spec.benchmark.add_iteration_duration(
                        iteration,
                        "hook_before_execute_tools_duration_ms",
                        (time.perf_counter() - started) * 1000,
                    )
                    spec.benchmark.add_span(
                        name="hook_before_execute_tools",
                        category="iteration_phase",
                        start_ms=before_tools_start_ms,
                        end_ms=before_tools_end_ms,
                        tid=10,
                        iteration=iteration,
                    )

                results, new_events, fatal_error, tools_wall_clock_duration_ms = await self._execute_tools(
                    spec, iteration, response.tool_calls,
                )
                tool_events.extend(new_events)
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                if spec.benchmark is not None:
                    spec.benchmark.set_tools_wall_clock_duration(iteration, tools_wall_clock_duration_ms)
                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    stop_reason = "tool_error"
                    context.error = error
                    context.stop_reason = stop_reason
                    if spec.benchmark is not None:
                        spec.benchmark.finalize_iteration(
                            iteration,
                            finish_reason=response.finish_reason,
                            stop_reason=stop_reason,
                            total_duration_ms=(time.perf_counter() - iteration_started) * 1000,
                        )
                    started = time.perf_counter()
                    after_iteration_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                    await hook.after_iteration(context)
                    after_iteration_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                    if spec.benchmark is not None:
                        spec.benchmark.add_iteration_duration(
                            iteration,
                            "hook_after_iteration_duration_ms",
                            (time.perf_counter() - started) * 1000,
                        )
                        spec.benchmark.add_span(
                            name="hook_after_iteration",
                            category="iteration_phase",
                            start_ms=after_iteration_start_ms,
                            end_ms=after_iteration_end_ms,
                            tid=10,
                            iteration=iteration,
                            args={"stop_reason": stop_reason},
                        )
                        iteration_end_ms = spec.benchmark.now_ms()
                        spec.benchmark.finalize_iteration(
                            iteration,
                            finish_reason=response.finish_reason,
                            stop_reason=stop_reason,
                            total_duration_ms=(time.perf_counter() - iteration_started) * 1000,
                        )
                        spec.benchmark.add_span(
                            name=f"iteration_{iteration}",
                            category="iteration",
                            start_ms=iteration_start_ms,
                            end_ms=iteration_end_ms,
                            tid=10,
                            iteration=iteration,
                            args={
                                "finish_reason": response.finish_reason,
                                "stop_reason": stop_reason,
                                "tool_count": len(response.tool_calls),
                            },
                        )
                    break

                started = time.perf_counter()
                tool_results_build_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                for tool_call, result in zip(response.tool_calls, results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    })
                tool_results_build_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                if spec.benchmark is not None:
                    spec.benchmark.add_iteration_duration(
                        iteration,
                        "message_build_tool_results_duration_ms",
                        (time.perf_counter() - started) * 1000,
                    )
                    spec.benchmark.add_span(
                        name="message_build_tool_results",
                        category="iteration_phase",
                        start_ms=tool_results_build_start_ms,
                        end_ms=tool_results_build_end_ms,
                        tid=10,
                        iteration=iteration,
                        args={"tool_count": len(results)},
                    )
                    spec.benchmark.finalize_iteration(
                        iteration,
                        finish_reason=response.finish_reason,
                        stop_reason="tool_calls",
                        total_duration_ms=(time.perf_counter() - iteration_started) * 1000,
                    )
                started = time.perf_counter()
                after_iteration_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                await hook.after_iteration(context)
                after_iteration_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                if spec.benchmark is not None:
                    spec.benchmark.add_iteration_duration(
                        iteration,
                        "hook_after_iteration_duration_ms",
                        (time.perf_counter() - started) * 1000,
                    )
                    spec.benchmark.add_span(
                        name="hook_after_iteration",
                        category="iteration_phase",
                        start_ms=after_iteration_start_ms,
                        end_ms=after_iteration_end_ms,
                        tid=10,
                        iteration=iteration,
                        args={"stop_reason": "tool_calls"},
                    )
                    iteration_end_ms = spec.benchmark.now_ms()
                    spec.benchmark.finalize_iteration(
                        iteration,
                        finish_reason=response.finish_reason,
                        stop_reason="tool_calls",
                        total_duration_ms=(time.perf_counter() - iteration_started) * 1000,
                    )
                    spec.benchmark.add_span(
                        name=f"iteration_{iteration}",
                        category="iteration",
                        start_ms=iteration_start_ms,
                        end_ms=iteration_end_ms,
                        tid=10,
                        iteration=iteration,
                        args={
                            "finish_reason": response.finish_reason,
                            "stop_reason": "tool_calls",
                            "tool_count": len(response.tool_calls),
                        },
                    )
                continue

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=False)

            started = time.perf_counter()
            finalize_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            clean = hook.finalize_content(context, response.content)
            finalize_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            if spec.benchmark is not None:
                spec.benchmark.add_iteration_duration(
                    iteration,
                    "finalize_content_duration_ms",
                    (time.perf_counter() - started) * 1000,
                )
                spec.benchmark.add_span(
                    name="finalize_content",
                    category="iteration_phase",
                    start_ms=finalize_start_ms,
                    end_ms=finalize_end_ms,
                    tid=10,
                    iteration=iteration,
                )
            if response.finish_reason == "error":
                final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                if spec.benchmark is not None:
                    spec.benchmark.finalize_iteration(
                        iteration,
                        finish_reason=response.finish_reason,
                        stop_reason=stop_reason,
                        total_duration_ms=(time.perf_counter() - iteration_started) * 1000,
                    )
                started = time.perf_counter()
                after_iteration_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                await hook.after_iteration(context)
                after_iteration_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
                if spec.benchmark is not None:
                    spec.benchmark.add_iteration_duration(
                        iteration,
                        "hook_after_iteration_duration_ms",
                        (time.perf_counter() - started) * 1000,
                    )
                    spec.benchmark.add_span(
                        name="hook_after_iteration",
                        category="iteration_phase",
                        start_ms=after_iteration_start_ms,
                        end_ms=after_iteration_end_ms,
                        tid=10,
                        iteration=iteration,
                        args={"stop_reason": stop_reason},
                    )
                    iteration_end_ms = spec.benchmark.now_ms()
                    spec.benchmark.finalize_iteration(
                        iteration,
                        finish_reason=response.finish_reason,
                        stop_reason=stop_reason,
                        total_duration_ms=(time.perf_counter() - iteration_started) * 1000,
                    )
                    spec.benchmark.add_span(
                        name=f"iteration_{iteration}",
                        category="iteration",
                        start_ms=iteration_start_ms,
                        end_ms=iteration_end_ms,
                        tid=10,
                        iteration=iteration,
                        args={
                            "finish_reason": response.finish_reason,
                            "stop_reason": stop_reason,
                            "tool_count": 0,
                        },
                    )
                break

            started = time.perf_counter()
            assistant_build_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            messages.append(build_assistant_message(
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            ))
            assistant_build_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            if spec.benchmark is not None:
                spec.benchmark.add_iteration_duration(
                    iteration,
                    "message_build_assistant_duration_ms",
                    (time.perf_counter() - started) * 1000,
                )
                spec.benchmark.add_span(
                    name="message_build_assistant",
                    category="iteration_phase",
                    start_ms=assistant_build_start_ms,
                    end_ms=assistant_build_end_ms,
                    tid=10,
                    iteration=iteration,
                    args={"has_tool_calls": False},
                )
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            if spec.benchmark is not None:
                spec.benchmark.finalize_iteration(
                    iteration,
                    finish_reason=response.finish_reason,
                    stop_reason=stop_reason,
                    total_duration_ms=(time.perf_counter() - iteration_started) * 1000,
                )
            started = time.perf_counter()
            after_iteration_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            await hook.after_iteration(context)
            after_iteration_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            if spec.benchmark is not None:
                spec.benchmark.add_iteration_duration(
                    iteration,
                    "hook_after_iteration_duration_ms",
                    (time.perf_counter() - started) * 1000,
                )
                spec.benchmark.add_span(
                    name="hook_after_iteration",
                    category="iteration_phase",
                    start_ms=after_iteration_start_ms,
                    end_ms=after_iteration_end_ms,
                    tid=10,
                    iteration=iteration,
                    args={"stop_reason": stop_reason},
                )
                iteration_end_ms = spec.benchmark.now_ms()
                spec.benchmark.finalize_iteration(
                    iteration,
                    finish_reason=response.finish_reason,
                    stop_reason=stop_reason,
                    total_duration_ms=(time.perf_counter() - iteration_started) * 1000,
                )
                spec.benchmark.add_span(
                    name=f"iteration_{iteration}",
                    category="iteration",
                    start_ms=iteration_start_ms,
                    end_ms=iteration_end_ms,
                    tid=10,
                    iteration=iteration,
                    args={
                        "finish_reason": response.finish_reason,
                        "stop_reason": stop_reason,
                        "tool_count": 0,
                    },
                )
            break
        else:
            stop_reason = "max_iterations"
            template = spec.max_iterations_message or _DEFAULT_MAX_ITERATIONS_MESSAGE
            final_content = template.format(max_iterations=spec.max_iterations)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            benchmark=spec.benchmark,
        )

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        iteration: int,
        tool_calls: list[ToolCallRequest],
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None, float]:
        started = time.perf_counter()
        tools_wall_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
        if spec.concurrent_tools:
            tool_results = await asyncio.gather(*(
                self._run_tool(spec, iteration, tool_call)
                for tool_call in tool_calls
            ))
        else:
            tool_results = [
                await self._run_tool(spec, iteration, tool_call)
                for tool_call in tool_calls
            ]
        tools_wall_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
        tools_wall_clock_duration_ms = (time.perf_counter() - started) * 1000
        if spec.benchmark is not None:
            spec.benchmark.add_span(
                name="tools_wall_clock",
                category="iteration_phase",
                start_ms=tools_wall_start_ms,
                end_ms=tools_wall_end_ms,
                tid=10,
                iteration=iteration,
                args={
                    "tool_count": len(tool_calls),
                    "concurrent": spec.concurrent_tools,
                },
            )

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error, tools_wall_clock_duration_ms

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        iteration: int,
        tool_call: ToolCallRequest,
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        started = time.perf_counter()
        tool_start_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
        try:
            result = await spec.tools.execute(tool_call.name, tool_call.arguments)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            tool_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if spec.benchmark is not None:
                duration_ms = (time.perf_counter() - started) * 1000
                spec.benchmark.add_tool_record(
                    iteration,
                    tool_call.name,
                    duration_ms,
                    "error",
                )
                spec.benchmark.add_span(
                    name=f"tool:{tool_call.name}",
                    category="tool",
                    start_ms=tool_start_ms,
                    end_ms=tool_end_ms,
                    tid=20,
                    iteration=iteration,
                    status="error",
                    args={
                        "arguments": tool_call.arguments,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None

        tool_end_ms = spec.benchmark.now_ms() if spec.benchmark is not None else 0.0
        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        status = "error" if isinstance(result, str) and result.startswith("Error") else "ok"
        if spec.benchmark is not None:
            duration_ms = (time.perf_counter() - started) * 1000
            spec.benchmark.add_tool_record(
                iteration,
                tool_call.name,
                duration_ms,
                status,
            )
            spec.benchmark.add_span(
                name=f"tool:{tool_call.name}",
                category="tool",
                start_ms=tool_start_ms,
                end_ms=tool_end_ms,
                tid=20,
                iteration=iteration,
                status=status,
                args={
                    "arguments": tool_call.arguments,
                    "detail": detail,
                },
            )
        return result, {
            "name": tool_call.name,
            "status": status,
            "detail": detail,
        }, None
