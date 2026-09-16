# DLSS Visual Enhancer for Synthetic Datasets

<img width="1752" height="694" alt="69xjanbxx8ph1" src="https://github.com/user-attachments/assets/f82a8118-b551-442e-8809-4a097c92aec5" />

A simplified, headless fork of [Merserk/dlss5-visual-enhancer](https://github.com/Merserk/dlss5-visual-enhancer), focused on enhancing synthetic images/videos for deep learning and computer vision workflows.

This fork runs directly from Python, with configuration in a YAML file and no web interface. 

## Why this fork?

Interactive previews are useful for individual images/videos, but a web interface can add time and friction when processing large datasets. This fork was motivated by workflows where managing large batches through the interface became time-consuming or encountered crashes.

The goal is a smaller, scriptable workflow: configure the runtime, point to a dataset, and write enhanced images/videos directly to disk. Removing the interface simplifies automation; it does not eliminate GPU memory limits or native runtime failures.

## Features

- Run image enhancement from a terminal or import it into a Python pipeline.
- Configure GPU, style, intensity, and processing passes in `config.yaml`.
- Process a folder of images/videos sequentially, saving each result immediately.
- Accept file paths, Pillow images, videos, or NumPy RGB/RGBA arrays through the Python API.

This is a Python entry point to the native Neuroframe engine, **not a pure-Python inference implementation**. It still requires the matching DLL runtime and compatible NVIDIA hardware. Gradio, PyTorch, and a separately installed CUDA toolkit are not required by these scripts.

## Requirements

- Windows 11, 64-bit, with Direct3D 12.
- Python 3.10 or newer, 64-bit.
- A compatible NVIDIA RTX GPU and driver.
- A complete Neuroframe runtime directory supporting **ABI version 6**.
- Python packages: `numpy`, `Pillow`, and `PyYAML`; `opencv-python`.

The wrapper checks for these files in the runtime directory:

- `neuroframe_engine.dll`
- `neuroframe_caller.dll`
- `nvngx_dlssnr.dll`

Use the complete `bin/runtime/dlssnr` directory from a matching [upstream release](https://github.com/Merserk/dlss5-visual-enhancer/releases), including its dependencies.

## Setup

Download or clone this fork, then open a terminal in its project directory. Keep `dlss_enhancer.py`, `main.py`, and `config.yaml` together.

Create a virtual environment and install the dependencies on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pillow pyyaml opencv-python
```

Edit `config.yaml` to point to your extracted runtime and dataset:

```yaml
dll_dir: "E:/DLSS-5-ENHANCER/bin/runtime/dlssnr"
input: "./input_images"
output: "./output_images"
gpu: 0
style: default
intensity: 2.0
passes: 1
```

Replace `dll_dir` with your actual runtime path. Use forward slashes for Windows paths, or single-quote paths containing backslashes.

| Setting | Meaning | Default if omitted |
| --- | --- | --- |
| `dll_dir` | Directory containing the matching native runtime | Required |
| `input` | Input file or folder for `dlss_enhancer.py`; input folder for `main.py` | Required for script workflows |
| `output` | Output directory | Required for script workflows |
| `gpu` | Nonnegative GPU index | `0` |
| `style` | `default`, `natural`, or `cinematic` | `default` |
| `intensity` | Enhancement intensity, from `0` to `2` | `1.0` |
| `passes` | Number of native enhancement passes, from `1` to `4` | `1` |

The supplied configuration uses intensity `2.0`; the fallback when the key is omitted is `1.0`.

## Process an image/video dataset

For large image/video folders, use the sequential workflow:

```powershell
.\.venv\Scripts\python.exe exe.py
```

## Use from Python

Enhance and save a single image:

```python
from dlss_enhancer import enhance_image

result = enhance_image("input_images/scene_001.png", config_path="config.yaml")
result.save("scene_001_enhanced.png")
```

Enhance a small batch:

```python
from pathlib import Path
from dlss_enhancer import enhance_images

paths = [Path("input_images/scene_001.png"), Path("input_images/scene_002.png")]
output = Path("output_images")
output.mkdir(exist_ok=True)

for source, result in zip(paths, enhance_images(paths, config_path="config.yaml")):
    result.save(output / f"{source.name}_enhanced.png")
```

## Acknowledgments

Credit to [Merserk](https://github.com/Merserk) for the original [DLSS 5 Visual Enhancer](https://github.com/Merserk/dlss5-visual-enhancer) and Neuroframe interface. This fork focuses on a reduced Python workflow for dataset preparation. The upstream project includes a broader interactive image and video toolset.
