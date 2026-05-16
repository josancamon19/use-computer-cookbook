# iOS sandbox reference

`IOSSandbox` is `Computer().create(type="ios", device_type=..., runtime=...)`. The sandbox is a backing CoreSimulator instance booted on a Mac mini. The simulator family (iPhone / iPad / Watch / TV / Vision) is encoded in `device_type`; if you omit it, the gateway picks an iPhone default.

## DSL surface

```python
# Apps
ios.apps.open_url("https://example.com")        # opens in the default browser
ios.apps.launch("com.apple.mobilesafari")       # by bundle id
ios.apps.terminate("com.apple.mobilesafari")
ios.apps.install("/path/to/app.app")            # path inside the simulator

# Input — coordinates are in the surface returned by ios.display.get_info()
ios.input.tap(x, y)
ios.input.long_press(x, y, duration=1.0)
ios.input.swipe(from_x, from_y, to_x, to_y)
ios.input.type_text("hello")                    # sends to the focused field
ios.input.press_button(Button.HOME)             # HOME, LOCK, VOLUME_UP/DOWN, SIRI
ios.input.press_key(Key.RETURN)                 # hardware key codes
ios.input.press_remote("select")                # tvOS remote controls

# Screen
ios.screenshot.take_full_screen()               # PNG bytes
ios.display.get_info()                          # device width/height in points
ios.accessibility.get_tree()                    # best-effort AX tree
```

`Button` and `Key` are enums in `use_computer.ios.input`. Strings work too — `ios.input.press_button("HOME")` is equivalent.

## Device families

When the user (or the cookbook YAML) creates an iOS sandbox, `device_type` and `runtime` decide what hardware is simulated:

```python
ios = cc.create(
    type="ios",
    device_type="com.apple.CoreSimulator.SimDeviceType.iPhone-17-Pro",
    runtime="com.apple.CoreSimulator.SimRuntime.iOS-26-4",
)
```

The runtime's OS family must match the device's family:

| Family       | Runtime family | What works                                                      |
| ------------ | -------------- | --------------------------------------------------------------- |
| iPhone       | `iOS`          | Everything: tap, long press, swipe, type, hardware buttons      |
| iPad         | `iOS`          | Same as iPhone, larger screen — tune coords against `display`   |
| Apple Watch  | `watchOS`      | Screenshot + best-effort tap/long press/swipe/button/key/launch |
| Apple TV     | `tvOS`         | Screenshot + `input.press_remote` + launch/key                  |
| Apple Vision | `visionOS`     | Screenshot + launch + tap/swipe against the 2D screenshot       |

Accessibility trees are best-effort for Watch/TV/Vision. Use them when
`available=True`; agents and collectors should fall back to screenshot-only
observations when AX is empty or unavailable.

To enumerate what the reservation's Mac mini actually has installed, the gateway exposes `/v1/platforms`. The cookbook's UI does this for you; from Python you can fetch it directly via `httpx` with the same bearer token.

## Sending screenshots to a vision model

iOS screenshots are larger than most CUA models accept (an iPhone 17 Pro screenshot is ~1206×2622 logical points and the raw PNG is even bigger). Always pipe them through `scale_screenshot_for_model` (in `runner.agents.base`) and multiply the model's coords by the returned `sx`, `sy` before calling `input.tap`:

```python
from runner.agents.base import scale_screenshot_for_model

png = ios.screenshot.take_full_screen()
api_bytes, api_w, api_h, sx, sy = scale_screenshot_for_model(png, "anthropic/claude-sonnet-4-6")
# model sees an api_w × api_h image and returns coordinate=[x, y]
ios.input.tap(int(x * sx), int(y * sy))
```

Kimi/Fireworks's vision cap is deliberately tight (896 px long-edge) because y-coord accuracy degrades on tall iOS shots past 1024 — `screenshot_cap_for_model` knows this. See `cookbook.md` for the per-model rationale.

## File transfer

iOS sandboxes share the same surface as macOS:

```python
ios.upload_bytes(b"hello\n", "/path/inside/sim/hello.txt")
ios.download_file("/path/inside/sim/hello.txt", "out.txt")
```

Paths are inside the simulator's data container. The simulator's filesystem is reachable via the gateway's file proxy; you cannot upload binaries that need to live on the host Mac. To install an app, use `apps.install()` with a `.app` path already on the host's filesystem (e.g. one that you uploaded to `/Users/lume/...` first via a sidecar macOS sandbox or staged at build time).

## Common patterns

**Open a URL and screenshot the loaded page:**

```python
with cc.create(type="ios") as ios:
    ios.apps.open_url("https://example.com")
    import time; time.sleep(2.0)
    png = ios.screenshot.take_full_screen()
```

**Type into a focused field (e.g. a search bar):**

```python
ios.input.tap(200, 60)                  # focus the field
ios.input.type_text("hello world")
ios.input.press_key(Key.RETURN)
```

**Pull-to-refresh:**

```python
info = ios.display.get_info()
cx = info.width // 2
ios.input.swipe(cx, info.height * 0.3, cx, info.height * 0.7)
```

**Home button:**

```python
from use_computer.ios.input import Button
ios.input.press_button(Button.HOME)
```

## Errors specific to iOS

- A simulator that hasn't finished booting will reject `input` and `apps` calls — `create()` waits for boot before returning, but if you keep a sandbox around and the simulator crashes, you'll see errors that look like the gateway is broken. Re-create the sandbox.
- `apps.open_url` with a non-`http(s)` URL silently routes via `xcrun simctl openurl` so custom URL schemes (`maps://`, `tel://`, an app's deep link) work.
- Calling `input.tap` on a non-iPhone/iPad family returns success but does nothing visible. There's no way for the SDK to tell you "this device doesn't have touch"; check `device_type` yourself before assuming taps will work.
