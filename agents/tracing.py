"""OpenTelemetry tracing helpers for agent components."""

from __future__ import annotations

import inspect
import json
from functools import wraps
from typing import Any, Callable

from agno.tools.file import FileTools
from agno.tools.mcp import MCPTools
from agno.tools.yfinance import YFinanceTools
from agno.learn.config import LearnedKnowledgeConfig, LearningMode
from agno.learn.stores.learned_knowledge import LearnedKnowledgeStore
from opentelemetry.trace import Status, StatusCode, get_tracer

tracer = get_tracer(__name__)

_MAX_ATTR = 8000


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:_MAX_ATTR]
    try:
        return json.dumps(value, default=str)[:_MAX_ATTR]
    except (TypeError, ValueError):
        return str(value)[:_MAX_ATTR]


def _record_call(span_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("input", _safe_str({"args": args, "kwargs": kwargs}))
        try:
            result = fn(*args, **kwargs)
            span.set_attribute("output", _safe_str(result))
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def trace_callable(span_name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("input", _safe_str({"args": args, "kwargs": kwargs}))
                try:
                    result = await fn(*args, **kwargs)
                    span.set_attribute("output", _safe_str(result))
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        return async_wrapper

    @wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _record_call(span_name, fn, *args, **kwargs)

    return sync_wrapper


def make_search_knowledge(knowledge: Any) -> Callable[[str], str]:
    def search_knowledge(query: str) -> str:
        with tracer.start_as_current_span("search_knowledge") as span:
            span.set_attribute("query", query)
            try:
                docs = knowledge.search(query=query)
                if not docs:
                    result = "No documents found"
                else:
                    result = json.dumps([doc.to_dict() for doc in docs], default=str)
                span.set_attribute("output", _safe_str(result))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    search_knowledge.__name__ = "search_knowledge_base"
    return search_knowledge


def make_learning_tools(store: Any, *, namespace: str = "global") -> tuple[Callable[..., str], Callable[..., str]]:
    raw_search = store._create_search_learnings_tool(namespace=namespace)
    raw_save = store._create_save_learning_tool(namespace=namespace)

    def search_learnings(query: str, limit: int = 5) -> str:
        with tracer.start_as_current_span("search_learnings") as span:
            span.set_attribute("query", query)
            span.set_attribute("limit", limit)
            try:
                result = raw_search(query=query, limit=limit)
                span.set_attribute("output", _safe_str(result))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def save_learning(
        title: str,
        learning: str,
        context: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        with tracer.start_as_current_span("save_learning") as span:
            span.set_attribute("title", title)
            span.set_attribute("learning", _safe_str(learning))
            span.set_attribute("context", _safe_str(context))
            span.set_attribute("tags", _safe_str(tags))
            try:
                result = raw_save(title=title, learning=learning, context=context, tags=tags)
                span.set_attribute("output", _safe_str(result))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    return search_learnings, save_learning


def setup_traced_learning(knowledge: Any, *, namespace: str = "global") -> tuple[LearnedKnowledgeStore, Callable[..., str], Callable[..., str]]:
    learned_config = LearnedKnowledgeConfig(
        mode=LearningMode.AGENTIC,
        namespace=namespace,
        knowledge=knowledge,
        agent_can_search=False,
        agent_can_save=False,
    )
    learned_store = LearnedKnowledgeStore(config=learned_config)
    search_learnings, save_learning = make_learning_tools(learned_store, namespace=namespace)
    return learned_store, search_learnings, save_learning


class TracedFileTools(FileTools):
    def read_file(self, file_name: str, encoding: str = "utf-8") -> str:
        with tracer.start_as_current_span("read_file") as span:
            span.set_attribute("file_name", file_name)
            try:
                result = super().read_file(file_name, encoding=encoding)
                span.set_attribute("output", _safe_str(result))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def list_files(self, **kwargs: Any) -> str:
        with tracer.start_as_current_span("list_files") as span:
            span.set_attribute("input", _safe_str(kwargs))
            try:
                result = super().list_files(**kwargs)
                span.set_attribute("output", _safe_str(result))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def search_files(self, pattern: str) -> str:
        with tracer.start_as_current_span("search_files") as span:
            span.set_attribute("pattern", pattern)
            try:
                result = super().search_files(pattern)
                span.set_attribute("output", _safe_str(result))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def save_file(self, contents: str, file_name: str, overwrite: bool = True, encoding: str = "utf-8") -> str:
        with tracer.start_as_current_span("save_file") as span:
            span.set_attribute("file_name", file_name)
            span.set_attribute("overwrite", overwrite)
            span.set_attribute("input", _safe_str(contents))
            try:
                result = super().save_file(contents, file_name, overwrite=overwrite, encoding=encoding)
                span.set_attribute("output", _safe_str(result))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise


def traced_yfinance_tools(**kwargs: Any) -> YFinanceTools:
    toolkit = YFinanceTools(**kwargs)
    for name, function in list(toolkit.functions.items()):
        entrypoint = function.entrypoint
        if entrypoint is None:
            continue

        def _wrap_tool(tool_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span("yfinance_tools") as span:
                    span.set_attribute("tool", tool_name)
                    span.set_attribute("input", _safe_str({"args": args, "kwargs": kwargs}))
                    try:
                        result = original(*args, **kwargs)
                        span.set_attribute("output", _safe_str(result))
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise

            return wrapper

        function.entrypoint = _wrap_tool(name, entrypoint)
    return toolkit


class TracedMCPTools(MCPTools):
    def register(self, function: Any, name: str | None = None) -> None:
        tool_name = name
        if tool_name is None:
            tool_name = function.name if hasattr(function, "name") else getattr(function, "__name__", "tool")
        if tool_name == "web_search_exa" and getattr(function, "entrypoint", None) is not None:
            function.entrypoint = trace_callable("web_search_exa", function.entrypoint)
        elif tool_name == "web_search_exa":
            function = trace_callable("web_search_exa", function)
        super().register(function, name=name)


def instrument_agent_entrypoint(agent: Any, name: str) -> None:
    for method_name in ("run", "arun"):
        original = getattr(agent, method_name, None)
        if original is None or getattr(original, "_otel_wrapped", False):
            continue

        if inspect.iscoroutinefunction(original):

            @wraps(original)
            async def async_run(*args: Any, _orig: Callable[..., Any] = original, _name: str = name, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(_name) as span:
                    request = kwargs.get("input") or kwargs.get("message") or (args[0] if args else "")
                    span.set_attribute("input", _safe_str(request))
                    span.set_attribute("agent.name", _name)
                    model = getattr(getattr(agent, "model", None), "id", None)
                    if model:
                        span.set_attribute("model", model)
                    try:
                        result = await _orig(*args, **kwargs)
                        span.set_attribute("output", _safe_str(getattr(result, "content", result)))
                        if metrics := getattr(result, "metrics", None):
                            span.set_attribute("token_usage", _safe_str(metrics))
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise

            async_run._otel_wrapped = True  # type: ignore[attr-defined]
            setattr(agent, method_name, async_run)
        else:

            @wraps(original)
            def sync_run(*args: Any, _orig: Callable[..., Any] = original, _name: str = name, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(_name) as span:
                    request = kwargs.get("input") or kwargs.get("message") or (args[0] if args else "")
                    span.set_attribute("input", _safe_str(request))
                    span.set_attribute("agent.name", _name)
                    model = getattr(getattr(agent, "model", None), "id", None)
                    if model:
                        span.set_attribute("model", model)
                    try:
                        result = _orig(*args, **kwargs)
                        span.set_attribute("output", _safe_str(getattr(result, "content", result)))
                        if metrics := getattr(result, "metrics", None):
                            span.set_attribute("token_usage", _safe_str(metrics))
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise

            sync_run._otel_wrapped = True  # type: ignore[attr-defined]
            setattr(agent, method_name, sync_run)


def instrument_agent_entrypoints(agents: list[tuple[Any, str]]) -> None:
    for agent, name in agents:
        instrument_agent_entrypoint(agent, name)
