"""Test: screenshot serialization under concurrent load.

Fires multiple screenshot requests to the same VM in parallel to measure
how the computer-server handles queuing. If screenshots serialize, N
concurrent requests take N * 2s instead of 2s.

Usage: uv run python tests/infra/test_screenshot_queue.py
"""

import asyncio
import time

from mmini.client import AsyncMmini

API = "http://10.10.10.2:8081"
API_KEY = "sk-dev-248ef4e90a1a3af4d3ff9542bb80098abef772719cb3b9f519eff481992a616e"


async def take_screenshot(sandbox, label: str) -> tuple[str, float, int]:
    start = time.monotonic()
    try:
        ss = await sandbox.screenshot.take_full_screen()
        elapsed = time.monotonic() - start
        return label, elapsed, len(ss)
    except Exception as e:
        elapsed = time.monotonic() - start
        return label, elapsed, -1


async def main():
    client = AsyncMmini(api_key=API_KEY, base_url=API)

    print("Creating sandbox...")
    sandbox = await client.create(type="macos", wait=True)
    print(f"Sandbox: {sandbox.sandbox_id} host={sandbox.host}")

    # Test 1: single screenshot baseline
    print("\n--- Test 1: single screenshot baseline ---")
    label, elapsed, size = await take_screenshot(sandbox, "single")
    print(f"  {elapsed:.2f}s ({size} bytes)")

    # Test 2: 3 concurrent screenshots to same VM
    print("\n--- Test 2: 3 concurrent screenshots ---")
    tasks = [take_screenshot(sandbox, f"concurrent-{i}") for i in range(3)]
    results = await asyncio.gather(*tasks)
    for label, elapsed, size in results:
        print(f"  {label}: {elapsed:.2f}s ({size} bytes)")
    max_time = max(r[1] for r in results)
    print(f"  Total wall time: {max_time:.2f}s (if serialized would be ~{len(results) * 2}s)")

    # Test 3: 5 concurrent screenshots
    print("\n--- Test 3: 5 concurrent screenshots ---")
    tasks = [take_screenshot(sandbox, f"concurrent-{i}") for i in range(5)]
    results = await asyncio.gather(*tasks)
    for label, elapsed, size in results:
        print(f"  {label}: {elapsed:.2f}s ({size} bytes)")
    max_time = max(r[1] for r in results)
    print(f"  Total wall time: {max_time:.2f}s (if serialized would be ~{len(results) * 2}s)")

    # Test 4: interleave screenshots with clicks
    print("\n--- Test 4: screenshot + click + screenshot rapid fire ---")
    start = time.monotonic()
    ss1 = await sandbox.screenshot.take_full_screen()
    t1 = time.monotonic() - start
    await sandbox.mouse.click(960, 540)
    t2 = time.monotonic() - start
    ss2 = await sandbox.screenshot.take_full_screen()
    t3 = time.monotonic() - start
    print(f"  screenshot1: {t1:.2f}s")
    print(f"  + click:     {t2:.2f}s")
    print(f"  + screenshot2: {t3:.2f}s")

    print("\nDestroying sandbox...")
    await sandbox.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
