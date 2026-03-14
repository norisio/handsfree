"""MCP client that bridges MCP servers with Gemini function calling.

Connects to MCP servers defined in mcp_config.json, converts their tools to
Gemini function declarations, and handles the tool call loop.
"""

import asyncio
import json
import os
import sys
import time

from google import genai
from google.genai import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Rate limiting for Gemini free tier
_RATE_LIMIT_INTERVAL = 4.0  # seconds between requests
_last_request_time = 0.0


def _rate_limit():
    """Enforce minimum interval between Gemini API requests."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < _RATE_LIMIT_INTERVAL:
        time.sleep(_RATE_LIMIT_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


class McpManager:
    """Manages connections to multiple MCP servers and their tools."""

    def __init__(self, config_path: str = "mcp_config.json"):
        self.config_path = config_path
        # Map tool_name → (server_name, session)
        self._tool_map: dict[str, tuple[str, ClientSession]] = {}
        # Gemini function declarations
        self._function_declarations: list[dict] = []
        # Async context managers to keep alive
        self._exit_stack: list = []

    @property
    def function_declarations(self) -> list[dict]:
        return self._function_declarations

    @property
    def has_tools(self) -> bool:
        return len(self._function_declarations) > 0

    async def connect(self) -> None:
        """Connect to all MCP servers and collect their tools."""
        if not os.path.exists(self.config_path):
            print(f"  No MCP config found at {self.config_path}, skipping.")
            return

        with open(self.config_path) as f:
            config = json.load(f)

        servers = config.get("mcpServers", {})
        for name, server_conf in servers.items():
            await self._connect_server(name, server_conf)

    async def _connect_server(self, name: str, conf: dict) -> None:
        """Connect to a single MCP server and register its tools."""
        params = StdioServerParameters(
            command=conf["command"],
            args=conf.get("args", []),
            env=conf.get("env"),
        )

        # Start the stdio client — we need to keep it alive
        ctx = stdio_client(params)
        streams = await ctx.__aenter__()
        self._exit_stack.append(ctx)

        session = ClientSession(*streams)
        session_ctx = session
        await session_ctx.__aenter__()
        self._exit_stack.append(session_ctx)

        await session.initialize()

        # List tools and register them
        result = await session.list_tools()
        for tool in result.tools:
            self._tool_map[tool.name] = (name, session)

            # Convert MCP tool schema → Gemini FunctionDeclaration
            params_schema = dict(tool.inputSchema) if tool.inputSchema else {"type": "object", "properties": {}}
            # Remove $schema key if present (Gemini doesn't accept it)
            params_schema.pop("$schema", None)

            self._function_declarations.append({
                "name": tool.name,
                "description": tool.description or "",
                "parameters": params_schema,
            })

            print(f"  Registered tool: {name}/{tool.name}")

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call an MCP tool by name and return the result as text."""
        if name not in self._tool_map:
            return f"Error: unknown tool '{name}'"

        server_name, session = self._tool_map[name]
        result = await session.call_tool(name, arguments)

        # Extract text from result content
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
        return "\n".join(parts) if parts else str(result)

    async def close(self) -> None:
        """Close all server connections."""
        for ctx in reversed(self._exit_stack):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._exit_stack.clear()
        self._tool_map.clear()
        self._function_declarations.clear()


def gemini_generate_with_tools(
    client: genai.Client,
    model: str,
    contents: list,
    system_instruction: str,
    mcp: McpManager,
    loop: asyncio.AbstractEventLoop,
) -> str:
    """Generate content with Gemini, handling tool calls via MCP.

    Runs the tool call loop synchronously (for integration with the
    non-async pipeline). Returns the final text response.
    """
    config_kwargs: dict = {"system_instruction": system_instruction}
    if mcp.has_tools:
        config_kwargs["tools"] = [types.Tool(function_declarations=mcp.function_declarations)]

    _rate_limit()
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config_kwargs,
    )

    # Tool call loop
    max_iterations = 5
    for _ in range(max_iterations):
        part = response.candidates[0].content.parts[0]

        if not part.function_call:
            break

        fn_call = part.function_call
        fn_name = fn_call.name
        fn_args = dict(fn_call.args) if fn_call.args else {}
        print(f"  [tool] {fn_name}({fn_args})")

        tool_result = loop.run_until_complete(mcp.call_tool(fn_name, fn_args))
        print(f"  [result] {tool_result}")

        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=fn_name, response={"result": tool_result})],
            )
        )

        _rate_limit()
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config_kwargs,
        )

    return response.text.strip()


def gemini_stream_with_tools(
    client: genai.Client,
    model: str,
    contents: list,
    system_instruction: str,
    mcp: McpManager,
    loop: asyncio.AbstractEventLoop,
):
    """Generate content with streaming, handling tool calls via MCP.

    Yields text chunks for streaming TTS. If the model requests tool calls,
    those are resolved via non-streaming round-trips first, then the final
    text response is yielded. If no tools are called, streams directly.
    """
    config_kwargs: dict = {"system_instruction": system_instruction}
    if mcp.has_tools:
        config_kwargs["tools"] = [types.Tool(function_declarations=mcp.function_declarations)]

    if not mcp.has_tools:
        # No tools: stream directly
        _rate_limit()
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config_kwargs,
        ):
            yield chunk.text or ""
        return

    # With tools: non-streaming first pass to check for tool calls
    _rate_limit()
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config_kwargs,
    )

    # Tool call loop
    max_iterations = 5
    for _ in range(max_iterations):
        part = response.candidates[0].content.parts[0]

        if not part.function_call:
            break

        fn_call = part.function_call
        fn_name = fn_call.name
        fn_args = dict(fn_call.args) if fn_call.args else {}
        print(f"  [tool] {fn_name}({fn_args})")

        tool_result = loop.run_until_complete(mcp.call_tool(fn_name, fn_args))
        print(f"  [result] {tool_result}")

        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=fn_name, response={"result": tool_result})],
            )
        )

        _rate_limit()
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config_kwargs,
        )

    # Yield the final text response
    yield response.text or ""
