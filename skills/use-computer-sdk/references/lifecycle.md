# Lifecycle, keepalive, retries, errors

The gateway aggressively reaps idle sandboxes. Long-running agents and slow think-time models trip this regularly, so the SDK ships a handful of knobs to keep sessions warm. The HTTP transport also retries idempotent calls under the hood, so understanding which failures actually surface to your code matters.

## The 2-minute idle reaper

Each sandbox is destroyed after 2 minutes of:

- no API calls (anything under `/v1/sandboxes/<id>/*` resets the timer), **and**
- no live SSH session on the VM, **and**
- no VNC connection (the dashboard's noVNC counts).

If any of those three is active, the sandbox is preserved. So a 10-minute `xcodebuild` over `exec_ssh` is fine — the SSH session keeps it alive even though `exec_ssh` itself only touches `LastActive` once at the start. A 10-minute _think_ by a slow LLM is not fine — there's no SSH, no API calls, the sandbox is gone.

## `start_keepalive` — the safety net

```python
with Computer().create() as mac:
    mac.start_keepalive(interval=30)        # daemon thread; idempotent on repeat calls
    # … long agent loop …
    mac.stop_keepalive()                    # optional; happens automatically on exit
```

The keepalive thread POSTs `/v1/sandboxes/<id>/keepalive` every `interval` seconds. Implementation note: it uses `urllib.request` directly (not the SDK's httpx client) so it doesn't share a connection pool with the caller and works equally well under sync `Sandbox` and `AsyncSandbox`. Don't push `interval` below 10s — you're just hammering the gateway.

Use it whenever:

- The agent does >2 min of think-time between actions (slow model, long reasoning chain).
- You're recording for >30s and downloading the recording afterwards (encoding can be slow).
- The user is driving the sandbox interactively via noVNC and you don't want to depend on their tab staying open.
- **You're using the sandbox by hand** — a debugging session in `ipython`, an emulator/REPL exploring an app, scripted actions with `time.sleep()` between them, anything where wall-clock seconds pass between API calls. The gateway only sees API calls, not what you're typing locally; turn on `start_keepalive` at the top of the session and forget about it.

You **don't** need it for normal SDK-driven scripts where you're polling screenshots every few seconds — those polls keep `LastActive` fresh on their own.

## Long-running exec

`exec_ssh(timeout=N)` — bump `N` for builds, large installs, etc.:

```python
mac.exec_ssh("xcodebuild -workspace App.xcworkspace -scheme App build", timeout=900)
```

The SSH connection itself keeps the sandbox alive (the gateway sees the SSH counter as non-zero from lume's per-VM heartbeat). The risk is the SSH connection dropping mid-exec for some other reason — in that case the activity counter goes to zero and the next reaper sweep kills the session. For really long jobs, pair `exec_ssh` with `start_keepalive(interval=30)` belt-and-suspenders.

## Auto-retries

The SDK's transport (`use_computer/retry.py`) retries on connection-level failures with exponential backoff for:

- `GET`, `HEAD`, `PUT`, `DELETE` (idempotent by definition).
- `POST /v1/sandboxes` (sandbox create) — explicit allow-list because that's the longest-tail call and worth retrying.

Other `POST`s (action dispatch, exec, file uploads of new bytes) are **not** retried — if they fail mid-flight, your code sees the error. If a retried call still fails after backoff you'll see `httpx.ConnectError` reach your code; that's a real gateway outage or a network partition, not a flake.

## Errors

All API-side failures inherit from `UseComputerError` (`use_computer.errors`):

```python
from use_computer.errors import UseComputerError, PlatformNotSupportedError

try:
    mac.exec_ssh("…")
except PlatformNotSupportedError:
    # tried a macOS verb on an iOS sandbox or vice versa
    ...
except UseComputerError as e:
    # API-side failure; e.status_code and e.error_code give structured detail
    ...
```

Common shapes you'll see:

- `unauthorized` (401): bad or wrong-env `mk_live_*` key. The key only validates against the env it was minted in — a prod key returns `unauthorized` against `api.dev.use.computer` and vice versa.
- `sandbox_not_found` (404): the sandbox was reaped (idle, or its reservation expired). Re-create.
- `reservation_expired` (403): the reservation hit `end_at`; no new sandboxes will be issued under that key until the user renews.
- `unavailable` (503): the warm pool is exhausted. Retry with backoff or wait for slots to free up.

## The 5-minute think-time gotcha

The single most common cause of "my agent randomly dies" bug reports:

1. Agent takes a screenshot.
2. Agent calls a slow model that takes >5 minutes to reason.
3. By the time the model returns an action, the sandbox is gone (the screenshot was the last API touch, ~5 min in the past).
4. The agent's next call returns `sandbox_not_found`.

Fix: either call `start_keepalive(interval=30)` once at sandbox creation, or send a heartbeat screenshot every minute from a sidecar thread. The cookbook's base agent does the keepalive call by default.

## Destroying explicitly

The context manager handles this for you. If you need to control destruction manually:

```python
sandbox = Computer().create()
try:
    # …
finally:
    sandbox.destroy()
```

`destroy()` is idempotent; calling it on an already-reaped sandbox returns successfully. The gateway's reaper picks up anything you forget after the 2-minute idle window.

## Where the lifecycle lives in the gateway

For the curious: `gateway/internal/session/store.go:IdleSessions()` is the reaper logic. Activity counters (`SSHSessions`, `VNCConnected`) are updated per-10s from the Mac mini via lume's heartbeat (`UpdateSlotActivity`), and reaper sweeps run every 30s. Reservation expiry runs on a separate 30s ticker (`internal/customerapi/reservation/expiry.go`) and is what handles the `end_at` deadline.
