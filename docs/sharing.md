# Share Inkforge

The Mac build is `dist/Inkforge.app`; distribute `dist/Inkforge-Darwin-arm64.zip`.
Extract it and open the app, then drop a GLB/glTF or USD scene onto the window.
The archive includes Python, Slang, OpenUSD, shaders and directional ink textures.
The recipient does not need Python, a virtual environment or this source tree.
Test scenes such as Bistro are not included; distribute assets with their own licenses.

The provided build targets Apple silicon and was exercised on macOS 26.6.2 / M5 Pro.
It is ad-hoc signed, not Developer ID signed or notarized. For normal public Mac
distribution, build with your Developer ID identity and notarize/staple the app
using your Apple developer account. No signing credentials are stored in this repo.

## Rebuild

```sh
uv sync --extra test --extra app
.venv/bin/python -m pytest -q
.venv/bin/python -m tools.build_app
.venv/bin/python -m tools.validate_app
```

`INKFORGE_CODESIGN_IDENTITY` optionally supplies the Developer ID identity to
PyInstaller. Build on each target OS and architecture: this is not a cross compiler.
The same script creates a standalone folder/ZIP on Windows or Linux; those packages
and rendering backends still need validation on actual target hardware. On Windows,
use `.venv\Scripts\python.exe` in the commands above. `validate_app` is Mac-specific.

The relocation check copies the entire app and two scenes into a temporary directory,
removes development Python environment variables, verifies all bundled native library
paths and the code signature, and renders both USD and GLB through the packaged executable.

## Controls

- Hold right mouse to look; WASD flies, Q/E moves down/up, Shift boosts speed.
- Wheel changes fly speed. Escape releases captured mouse, then exits.
- F1 shows/hides the resizable settings panel.
- F2 shows/hides the independent performance overlay (also available in Camera and diagnostics).
  It shows live FPS, average/p95 frame time, CPU encode/submit time, acquire/present
  waits and render resolution over a rolling two-second window. These are wall-clock
  timings, including presentation pacing, not GPU timestamps.
- T / Shift+T smoothly advances/reverses two hours. U pauses/resumes the day cycle.
- C pauses/resumes clouds, L cycles lighting mode, G cycles ground mode.
- P saves a clean PNG and camera/style JSON in `~/Pictures/Inkforge`.

Restore a captured view with `Inkforge --session capture.json --viewer`.
Pass `--output capture.png` instead of `--viewer` for an offscreen render.

## Native host reference

`ntc_viewer` uses C++20, GLFW, ImGui and Slang/slang-rhi for its render loop. Its
USD importer still invokes Python. Its current bundle uses source/build-tree paths
and is not relocatable. Inkforge adopts the fly-control conventions but retains its
Python/SlangPy host. A future C++ host could reuse these Slang shaders; packaging
the current host does not imply that migration has occurred.

Build tooling: [PyInstaller usage](https://pyinstaller.org/en/stable/usage.html)
and [macOS signing notes](https://pyinstaller.org/en/stable/feature-notes.html).
Dependency license notices are retained in the bundled package metadata.
