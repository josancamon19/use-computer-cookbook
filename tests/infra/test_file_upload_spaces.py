"""Test: file upload/download with spaces in paths.

Verifies that upload_dir, upload, download_file, and exec_ssh
all handle paths with spaces correctly.

Usage: uv run python tests/infra/test_file_upload_spaces.py
"""

import asyncio
import tempfile
from pathlib import Path

from mmini.client import AsyncMmini

API = "http://10.10.10.2:8081"
API_KEY = "sk-dev-248ef4e90a1a3af4d3ff9542bb80098abef772719cb3b9f519eff481992a616e"


async def main():
    client = AsyncMmini(api_key=API_KEY, base_url=API)

    print("Creating sandbox...")
    sandbox = await client.create(type="macos", wait=True)
    print(f"Sandbox: {sandbox.sandbox_id} host={sandbox.host}")

    # Test 1: upload single file with spaces in remote path
    print("\n--- Test 1: upload file to path with spaces ---")
    try:
        await sandbox.upload_bytes(b"hello spaces", "/tmp/My Documents/test file.txt")
        result = await sandbox.exec_ssh('cat "/tmp/My Documents/test file.txt"')
        assert result.stdout.strip() == "hello spaces", f"got: {result.stdout}"
        print("  PASS: single file with spaces")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 2: upload directory with files that have spaces
    print("\n--- Test 2: upload_dir with spaces in filenames ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "Language 1.txt").write_text("hello")
        (tmpdir / "Matterhorn Man.md").write_text("# test")
        (tmpdir / "sub dir").mkdir()
        (tmpdir / "sub dir" / "nested file.txt").write_text("nested")

        try:
            await sandbox.upload_dir(tmpdir, "/tmp/test_spaces")
            result = await sandbox.exec_ssh('cat "/tmp/test_spaces/Language 1.txt"')
            assert result.stdout.strip() == "hello", f"got: {result.stdout}"
            result = await sandbox.exec_ssh('cat "/tmp/test_spaces/sub dir/nested file.txt"')
            assert result.stdout.strip() == "nested", f"got: {result.stdout}"
            print("  PASS: directory with spaces")
        except Exception as e:
            print(f"  FAIL: {e}")

    # Test 3: download file with spaces
    print("\n--- Test 3: download file with spaces ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            await sandbox.download_file("/tmp/My Documents/test file.txt", f"{tmpdir}/downloaded.txt")
            content = Path(f"{tmpdir}/downloaded.txt").read_text()
            assert content == "hello spaces", f"got: {content}"
            print("  PASS: download file with spaces")
        except Exception as e:
            print(f"  FAIL: {e}")

    # Test 4: exec with paths containing spaces
    print("\n--- Test 4: exec with space paths ---")
    try:
        result = await sandbox.exec_ssh('ls -la "/tmp/My Documents/"')
        assert "test file.txt" in result.stdout, f"got: {result.stdout}"
        print("  PASS: exec with space paths")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 5: the actual benchmark_files pattern
    print("\n--- Test 5: simulate Benchmark_Backup upload ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "NCE").mkdir()
        (tmpdir / "NCE" / "Language 1.txt").write_text("test lang")
        (tmpdir / "Matterhorn Man.docx").write_bytes(b"fake docx")

        try:
            await sandbox.upload_dir(tmpdir, "/Users/lume/Benchmark_Backup/benchmark_files")
            result = await sandbox.exec_ssh('cat "/Users/lume/Benchmark_Backup/benchmark_files/NCE/Language 1.txt"')
            assert result.stdout.strip() == "test lang", f"got: {result.stdout}"
            print("  PASS: Benchmark_Backup pattern")
        except Exception as e:
            print(f"  FAIL: {e}")

    print("\nDestroying sandbox...")
    await sandbox.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
