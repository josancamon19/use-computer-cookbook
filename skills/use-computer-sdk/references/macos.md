# macOS sandbox reference

`MacOSSandbox` is what you get from `Computer().create(type="macos")`. The sandbox is a real macOS VM cloned from a base image on a Mac mini, with SSH to the `lume` user already wired up and a small HTTP control plane proxied through the gateway.

## DSL surface

```python
# Shell access — the catch-all for anything the DSL doesn't cover
mac.exec_ssh(cmd, timeout=120)          # → ExecResult(stdout, stderr, exit_code)

# Keyboard
mac.keyboard.type(text)                 # delay= per keystroke (ms)
mac.keyboard.press("Return")            # also: modifiers=["cmd"]
mac.keyboard.hotkey("cmd+s")            # parses a literal chord

# Mouse — coordinates are in display points, not physical pixels
mac.mouse.click(x, y, button="left", double=False)
mac.mouse.move(x, y)
mac.mouse.drag(from_x, from_y, to_x, to_y, button="left")
mac.mouse.scroll(x, y, direction="down", amount=3)
mac.mouse.get_position()                # → CursorPosition(x, y)

# Screen
mac.screenshot.take_full_screen()       # → PNG bytes
mac.screenshot.take_region(x, y, w, h)  # PNG bytes of a crop
mac.display.get_info()                  # → DisplayInfo(width, height, scale_factor)
```

## SSH for everything else

Anything you can't express through the DSL (filesystem queries, opening apps with arguments, AppleScript, shell pipelines) — do it over `exec_ssh`. The default user is `lume`, sudo is configured, and the VM is wiped at the end of the session, so don't be precious about installs.

```python
r = mac.exec_ssh("osascript -e 'tell application \"Safari\" to activate'")
if r.exit_code != 0:
    raise RuntimeError(r.stderr)

mac.exec_ssh("xcodebuild -workspace App.xcworkspace -scheme App build", timeout=900)
```

`exec_ssh` only touches the session timestamp at the start of the call, so a 10-minute build doesn't keep heartbeating on its own. See `lifecycle.md` for how to combine long execs with keepalive.

### Caveat: AppleScript that walks the Accessibility tree

SSH is the right hammer for most catch-all needs, but **AppleScript that touches the Accessibility tree** (`tell application "System Events" to …`, `attribute "AX..." of …`, `keystroke`) silently fails over `exec_ssh`. macOS's TCC tracks the responsibility chain of the calling process; SSH puts `sshd-keygen-wrapper` in that chain, which is not pre-approved for Accessibility, so AX reads return `-25211` and the script alarms-out.

The SDK ships a transpiler that rewrites these patterns into calls against an in-VM helper (`/usr/local/bin/ax_helper.py`, baked into the base image) routed through the CUA server's `/cmd` endpoint — that path's responsibility chain (launchd → cua-server → bash → python3.12) is what the TCC grant actually applies to. Use it before piping a verifier script over `exec_ssh`:

```python
from use_computer.ax_transpile import transpile, patch_curl_timeouts, needs_exec_ax

raw = open("tests/test.sh").read()
if needs_exec_ax(raw):
    rewritten, _ = transpile(raw)
    rewritten, _ = patch_curl_timeouts(rewritten)
    mac.upload_bytes(rewritten.encode(), "/tmp/test.sh")
    r = mac.exec_ssh("bash /tmp/test.sh")
else:
    r = mac.exec_ssh(raw)
```

`transpile()` recognizes six AppleScript shapes (attribute reads, dock items, `keystroke`, etc.) — see the module docstring in `sdk/use_computer/ax_transpile.py` for the exact coverage. Lines that don't match a known pattern pass through unchanged, so it's safe to apply over any script. `patch_curl_timeouts()` is the companion that wraps long `curl` calls in `alarm()` so a wedged endpoint can't hang a grader.

This is what the cookbook's macOSWorld adapter does to every benchmark's `pre_command` and `tests/test.sh` before uploading them into the VM — see `cookbook.md` for the full pipeline.

## File transfer

```python
mac.upload("local.txt", "/Users/lume/Desktop/local.txt")          # path → path
mac.upload_bytes(b"hello\n", "/Users/lume/Desktop/hello.txt")     # bytes → path
mac.download_file("/Users/lume/Desktop/hello.txt", "out.txt")

mac.upload_dir("./assets", "/Users/lume/Desktop/assets")           # tars + extracts
mac.download_dir("/Users/lume/Desktop/results", "./results")
```

Parent directories on the remote side must exist — `upload_bytes` won't `mkdir -p` for you. Either call `exec_ssh("mkdir -p ...")` first or use `upload_dir`, which packages a tarball server-side and unpacks atomically.

## Coordinate system

`screenshot.take_full_screen()` can return a retina-resolution image (2x the logical display). `mouse.click(x, y)` takes logical points. Always size your click coordinates against `display.get_info().width / .height`, not the PNG's raw dimensions, or every click will land at half the intended location.

```python
info = mac.display.get_info()
center_x, center_y = info.width // 2, info.height // 2
mac.mouse.click(center_x, center_y)
```

### Sending screenshots to a vision model

If you're feeding the screenshot to a CUA model (Claude, OpenAI, Gemini, Kimi/Fireworks, …), the model returns coordinates in *its own resized-image space*, not your native screenshot. The cookbook ships `runner.agents.base.scale_screenshot_for_model(...)` to do the resize client-side and hand you the inverse scale factors:

```python
from runner.agents.base import scale_screenshot_for_model

png = mac.screenshot.take_full_screen()
api_bytes, api_w, api_h, sx, sy = scale_screenshot_for_model(png, "anthropic/claude-sonnet-4-6")
# send api_bytes to the model; model returns coordinate=[x, y]
mac.mouse.click(int(x * sx), int(y * sy))
```

The per-model registry lives in `screenshot_cap_for_model(model)` (also in `runner.agents.base`) — change there, not at call sites. See `cookbook.md` for the full rationale. If you're consuming the SDK *outside* this cookbook, you can copy these two helpers (~40 lines) into your own project; they don't depend on anything in `runner.agents`.

## Common patterns

**Open an app and wait for the window:**

```python
mac.exec_ssh("open -a Safari")
# Brief settle — Safari needs a moment after launch
import time; time.sleep(1.0)
png = mac.screenshot.take_full_screen()
```

**Cmd+Tab between apps:**

```python
mac.keyboard.press("Tab", modifiers=["cmd"])     # equivalent to hotkey("cmd+tab")
```

**Drag a file onto the dock:**

```python
mac.mouse.drag(from_x=100, from_y=200, to_x=800, to_y=1100, button="left")
```

**AppleScript for app state graders:**

```python
r = mac.exec_ssh(
    'osascript -e \'tell application "Numbers" '
    'to get value of cell "B2" of table 1 of sheet 1 of document 1\''
)
print(r.stdout.strip())     # → "42"
```

The accessibility tree is reachable the same way via `tell application "System Events"`. Apple's [AppleScript Language Guide](https://developer.apple.com/library/archive/documentation/AppleScript/Conceptual/AppleScriptLangGuide/) has the full vocabulary.

## Errors specific to macOS

- `exec_ssh` returns a non-zero `exit_code` rather than raising — branch on it explicitly.
- If you see "permission denied" from `exec_ssh`, you probably tried to write to a system path; `~/Documents` or `~/Desktop` are safe defaults.
- `mouse.click` on coordinates outside the display silently no-ops on macOS. Verify against `display.get_info()` before falling down a "why isn't the click registering" hole.
