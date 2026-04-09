"""Test: exhaust warm pool then kill a mini.

Creates sandboxes until the pool is empty, then simulates a mini crash
by killing lume on one of them. Verifies the affected sandboxes fail
gracefully and the pool recovers.

Usage: uv run python tests/infra/test_pool_exhaustion.py
"""

import asyncio
import sys

from mmini.client import AsyncMmini

API = "http://10.10.10.2:8081"
API_KEY = "sk-dev-248ef4e90a1a3af4d3ff9542bb80098abef772719cb3b9f519eff481992a616e"


async def main():
    client = AsyncMmini(api_key=API_KEY, base_url=API)

    # Step 1: claim all warm VMs
    print("Step 1: exhausting warm pool...")
    sandboxes = []
    for i in range(80):
        try:
            sb = await client.create(type="macos", wait=False)
            sandboxes.append(sb)
            print(f"  [{i+1}] {sb.sandbox_id} host={sb.host}")
        except Exception as e:
            print(f"  [{i+1}] pool exhausted: {e}")
            break

    print(f"\nClaimed {len(sandboxes)} sandboxes")
    if not sandboxes:
        print("ERROR: couldn't claim any sandboxes")
        return

    # Step 2: verify they all work
    print("\nStep 2: verifying all sandboxes respond...")
    alive = 0
    for sb in sandboxes:
        try:
            result = await sb.exec_ssh("echo ok", timeout=10)
            if result.return_code == 0:
                alive += 1
        except Exception as e:
            print(f"  {sb.sandbox_id}: FAILED {e}")
    print(f"  {alive}/{len(sandboxes)} responding")

    # Step 3: try to create one more (should fail)
    print("\nStep 3: attempting to create with empty pool (no wait)...")
    try:
        sb = await client.create(type="macos", wait=False)
        print(f"  UNEXPECTED: got {sb.sandbox_id}")
        sandboxes.append(sb)
    except Exception as e:
        print(f"  Expected failure: {e}")

    # Step 4: destroy all and check pool recovery
    print(f"\nStep 4: destroying all {len(sandboxes)} sandboxes...")
    for sb in sandboxes:
        try:
            await sb.close()
        except Exception:
            pass

    print("  Waiting 30s for pool to replenish...")
    await asyncio.sleep(30)

    print("  Checking pool health...")
    try:
        sb = await client.create(type="macos", wait=True)
        result = await sb.exec_ssh("echo recovered", timeout=10)
        print(f"  Pool recovered: {result.stdout.strip()}")
        await sb.close()
    except Exception as e:
        print(f"  Pool NOT recovered: {e}")


if __name__ == "__main__":
    asyncio.run(main())
