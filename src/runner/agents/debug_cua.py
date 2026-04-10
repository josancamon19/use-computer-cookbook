"""Debug CUA agent — smoke-tests infra without calling any LLM."""

from __future__ import annotations

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from runner.agents.base_cua import BaseCUAAgent


class DebugCUAAgent(BaseCUAAgent):
    @staticmethod
    def name() -> str:
        return "debug-cua"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        sandbox = await self.pre_run(environment)
        ss = await sandbox.screenshot.take_full_screen()
        await sandbox.mouse.click(960, 540)
        result = await sandbox.exec_ssh("echo hello world")
        print(f"[debug] screenshot={len(ss)} bytes, exec={result}")
        await self._fire_in_process(environment, 1)  # after step 1
        await self.post_run(context, "debug", "debug-cua")
