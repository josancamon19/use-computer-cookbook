# adapters/

Converters that turn external task sources into the [Harbor task format](https://harborframework.com/docs/tasks) so they can be run with `harbor run`.

| Adapter       | Source                                                                   | CLI                                                                                                                     |
| ------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `adhoc/`      | Hand-written prompt JSON (`tasks/{ios,macos}.json`)                      | `uv run python -m runner.adapters.adhoc.export <tasks.json>`                                                            |
| `collected/`  | Tasks recorded via the use.computer `/collect` UI on the gateway         | `uv run python -m runner.adapters.collected.adapter --all [--platform macos\|ios]`                                      |
| `macosworld/` | The [macOSWorld](https://macos-world.github.io) benchmark repo | `uv run python -m runner.adapters.macosworld.run_adapter --macosworld-root <path> --task-dir datasets/macosworld_ready` |

All three write their output as a directory of Harbor task dirs (each containing `instruction.md`, `task.toml`, optionally `tests/`, `actions.json`, etc.). Point `harbor run -p <dir>` at the result.
