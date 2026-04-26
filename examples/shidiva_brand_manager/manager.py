from __future__ import annotations

import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents import Runner, gen_trace_id, trace

from .agents.orchestrator_agent import super_producer_agent


class ShidivaBrandManager:
    """Orchestrates the Super Producer agent system for Shidiva brand management."""

    def __init__(self) -> None:
        self.console = Console()

    async def run(self, request: str) -> None:
        trace_id = gen_trace_id()

        self.console.print(
            Panel.fit(
                "[bold magenta]✦ SHIDIVA BRAND MANAGER — SUPER PRODUCER ✦[/bold magenta]\n"
                "[dim]Powered by the world's strongest AI · Digital · Human intelligence[/dim]",
                border_style="magenta",
            )
        )
        self.console.print(
            f"[dim]Trace: https://platform.openai.com/traces/trace?trace_id={trace_id}[/dim]\n"
        )

        with trace("Shidiva Brand Manager", trace_id=trace_id):
            result = await self._run_super_producer(request)

        self.console.print("\n")
        self.console.print(
            Panel(
                Markdown(result),
                title="[bold magenta]✦ SHIDIVA BRAND STRATEGY — SUPER PRODUCER OUTPUT ✦[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )

    async def _run_super_producer(self, request: str) -> str:
        phases = [
            "Activating Super Producer intelligence...",
            "Deploying market intelligence systems...",
            "Analyzing global competitive landscape...",
            "Building digital growth architecture...",
            "Crafting brand strategy framework...",
            "Synthesizing world-class brand plan...",
            "Finalizing Shidiva brand strategy...",
        ]

        with Progress(
            SpinnerColumn(style="magenta"),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task(phases[0], total=None)

            result_container: list[str] = []
            phase_index = 0
            last_update = time.time()

            async def _stream() -> None:
                nonlocal phase_index, last_update
                stream = Runner.run_streamed(
                    super_producer_agent,
                    f"Brand management request for Shidiva:\n\n{request}",
                )
                async for _ in stream.stream_events():
                    if time.time() - last_update > 6 and phase_index < len(phases) - 1:
                        phase_index += 1
                        progress.update(task, description=phases[phase_index])
                        last_update = time.time()
                result_container.append(str(stream.final_output or ""))

            await _stream()

        return result_container[0] if result_container else ""
