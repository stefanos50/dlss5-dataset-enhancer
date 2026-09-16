#!/usr/bin/env python3
"""Standalone image enhancement through Merserk's Neuroframe ABI v6 DLL.

Requirements: 64-bit Python 3.10+, Windows 11, compatible NVIDIA RTX/driver,
              pip install numpy pillow pyyaml
Use the complete bin/runtime/dlssnr directory from a matching packaged release.
No Gradio, repository imports, PyTorch, or CUDA toolkit required.

Usage:
  python enhance_images.py  # Reads config.yaml beside this module; no CLI arguments.
  from enhance_images import enhance_images
  results = enhance_images(["photo.jpg"], config_path="config.yaml")
  results[0].save("enhanced.png")

The public functions accept paths, Pillow images, or uint8 RGB/RGBA NumPy arrays.
They return new Pillow images and never save files. OpenCV BGR arrays must first
be converted to RGB. Relative paths in YAML resolve from its containing folder;
image paths passed to the functions resolve from the working directory.
Importing this module does not read YAML or initialize the native engine.

Outputs same-size 8-bit sRGB PNGs. First frame/page only; EXIF is not copied.
The native engine does inference on the GPU using host staging buffers.
Native inference has not been tested in the environment that created this file.
Interface reference: Merserk/dlss5-visual-enhancer commit
c95c050ead79b9409f140e0b2a66a7b91cffb258, src/core/neural_bridge.py.
"""
from __future__ import annotations

import ctypes as C
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import threading

import yaml

import numpy as np
from PIL import Image, ImageCms, ImageOps

# ABI layout must match the engine exactly; use native Windows x64 alignment.
class RenderParametersV6(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
        ("style", C.c_int32), ("intensity", C.c_float),
        ("tone", C.c_float), ("structure", C.c_float),
        ("skin", C.c_float), ("automask", C.c_int32),
        ("reset", C.c_int32), ("color_strength", C.c_float),
        ("tone_preservation", C.c_float), ("mask_memory_type", C.c_uint32),
        ("mask_width", C.c_uint32), ("mask_height", C.c_uint32),
        ("mask_stride", C.c_uint32), ("mask_plane", C.c_uint64),
        ("face_skin_protection", C.c_float), ("grain_preservation", C.c_float),
        ("nr_passes", C.c_int32), ("shimmer_suppression", C.c_float),
        ("prefer_nvof", C.c_int32),
    ]


class Engine:
    def __init__(self, directory: Path, gpu: int):
        if sys.platform != "win32" or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError("Requires 64-bit Python on Windows and an NVIDIA RTX GPU.")
        directory = directory.resolve(strict=True)
        for name in ("neuroframe_engine.dll", "neuroframe_caller.dll", "nvngx_dlssnr.dll"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Missing {directory / name}. Use the matching release runtime.")
        # Keep this handle alive for delayed dependency loads.
        self.dll_search = os.add_dll_directory(str(directory))
        self.dll = C.WinDLL(str(directory / "neuroframe_engine.dll"))
        dll = self.dll
        try:
            dll.dlss5nr_frame_abi_version.argtypes = []
            dll.dlss5nr_frame_abi_version.restype = C.c_uint32
            abi = dll.dlss5nr_frame_abi_version()
            if abi != 6:
                raise RuntimeError(f"This script needs ABI 6; your DLL reports ABI {abi}.")
            dll.dlss5nr_init.argtypes = [C.c_int, C.c_wchar_p, C.c_char_p, C.c_int]
            dll.dlss5nr_init.restype = C.c_int
            dll.dlss5nr_process_v6.argtypes = [
                C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_int, C.c_int,
                C.POINTER(RenderParametersV6), C.c_char_p, C.c_int,
            ]
            dll.dlss5nr_process_v6.restype = C.c_int
        except AttributeError as exc:
            raise RuntimeError("DLL is missing required Neuroframe ABI v6 exports.") from exc
        error = C.create_string_buffer(4096)
        if not dll.dlss5nr_init(gpu, str(directory), error, len(error)):
            raise RuntimeError("DLL initialization failed: " + error.value.decode("utf-8", "replace"))
        # Do not unload the DLL or invoke NGX shutdown: upstream reports hangs.
        # Let process exit dispose of the runtime once the batch is complete.

    def process(self, rgb: np.ndarray, params: RenderParametersV6) -> np.ndarray:
        source = np.ascontiguousarray(rgb, dtype=np.float32) / np.float32(255.0)
        output = np.empty_like(source)
        height, width, _ = source.shape
        error = C.create_string_buffer(4096)
        params.reset = 1  # Every still image is an independent scene.
        ok = self.dll.dlss5nr_process_v6(
            source.ctypes.data_as(C.POINTER(C.c_float)),
            output.ctypes.data_as(C.POINTER(C.c_float)), width, height,
            C.byref(params), error, len(error),
        )
        if not ok:
            raise RuntimeError("Native inference failed: " + error.value.decode("utf-8", "replace"))
        if not np.isfinite(output).all():
            raise RuntimeError("Native engine returned non-finite pixel values.")
        return (np.clip(output, 0, 1) * 255).astype(np.uint8)


def _image_pixels(value):
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as original:
            return _image_pixels(original)
    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] not in (3, 4):
            raise ValueError("NumPy images must have dtype uint8 and shape (H, W, 3 or 4), in RGB/RGBA order.")
        value = Image.fromarray(value)
    if not isinstance(value, Image.Image):
        raise TypeError("Each image must be a path, Pillow image, or uint8 RGB/RGBA array.")
    image = ImageOps.exif_transpose(value)
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    profile = value.info.get("icc_profile")
    if profile:
        color = image if image.mode in ("RGB", "CMYK", "L", "LAB") else rgba.convert("RGB")
        rgb = ImageCms.profileToProfile(
            color, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
            ImageCms.createProfile("sRGB"), outputMode="RGB",
        )
    else:
        rgb = rgba.convert("RGB")
    return np.array(rgb), alpha


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
_ENGINE = None
_ENGINE_KEY = None
_ENGINE_ERROR = None
_ENGINE_LOCK = threading.RLock()


def load_config(path=DEFAULT_CONFIG) -> SimpleNamespace:
    path = Path(path).expanduser().resolve(strict=True)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("YAML configuration must be a mapping of setting names to values.")
    allowed = {"dll_dir", "input", "output", "gpu", "style", "intensity", "passes"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(map(str, unknown))}")
    for key in ("dll_dir", "input", "output"):
        if key != "dll_dir" and key not in config:
            continue
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Configuration '{key}' must be a nonempty path string.")
        value = Path(value).expanduser()
        config[key] = value if value.is_absolute() else path.parent / value
    for key, default in {"gpu": 0, "style": "default", "intensity": 1.0, "passes": 1}.items():
        config.setdefault(key, default)
    if type(config["gpu"]) is not int or config["gpu"] < 0:
        raise ValueError("'gpu' must be a nonnegative integer.")
    if config["style"] not in ("default", "natural", "cinematic"):
        raise ValueError("'style' must be default, natural, or cinematic.")
    if type(config["intensity"]) not in (int, float) or not 0 <= config["intensity"] <= 2:
        raise ValueError("'intensity' must be a number between 0 and 2.")
    if type(config["passes"]) is not int or not 1 <= config["passes"] <= 4:
        raise ValueError("'passes' must be an integer between 1 and 4.")
    return SimpleNamespace(**config)


def _parameters(config):
    params = RenderParametersV6()
    params.struct_size = C.sizeof(params)
    params.abi_version = 6
    params.style = {"default": 0, "natural": 1, "cinematic": 2}[config.style]
    params.intensity = config.intensity
    params.tone = params.structure = params.color_strength = 1.0
    params.skin = -1.0
    params.mask_memory_type = 2  # MEMORY_NONE
    params.nr_passes = config.passes
    return params


def _enhance_one(image, config):
    global _ENGINE, _ENGINE_KEY, _ENGINE_ERROR
    rgb, alpha = _image_pixels(image)
    h, w = rgb.shape[:2]
    if not (64 <= w <= 7680 and 64 <= h <= 4320):
        raise ValueError(f"{w}x{h} is outside 64x64..7680x4320; resize first.")
    key = (config.dll_dir.resolve(), config.gpu)
    # NGX is process-wide. Serialize initialization and inference across callers.
    with _ENGINE_LOCK:
        if _ENGINE_ERROR is not None:
            raise RuntimeError("The native engine previously failed; restart this process.") from _ENGINE_ERROR
        if _ENGINE_KEY is not None and _ENGINE_KEY != key:
            raise RuntimeError("Changing DLL directory or GPU requires a new Python process.")
        try:
            if _ENGINE is None:
                _ENGINE = Engine(*key)
                _ENGINE_KEY = key
            enhanced = _ENGINE.process(rgb, _parameters(config))
        except Exception as exc:
            _ENGINE_ERROR = exc
            raise
    result = Image.fromarray(np.dstack((enhanced, alpha)))
    result.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return result


def enhance_images(images, config_path=DEFAULT_CONFIG) -> list[Image.Image]:
    """Return enhanced RGBA Pillow images in input order, without saving.

    images: one path/Pillow image/uint8 RGB(A) array, or an iterable of them.
    Always returns a list (an empty iterable returns []). The caller owns the
    results. Input images are not modified or closed. The engine is reused
    between calls; style/intensity/passes are reread from YAML each call.
    Native failures propagate as exceptions; no partial result list is returned.
    """
    config = load_config(config_path)
    if isinstance(images, (str, os.PathLike, Image.Image, np.ndarray)):
        images = [images]
    return [_enhance_one(image, config) for image in images]


def enhance_image(image, config_path=DEFAULT_CONFIG) -> Image.Image:
    """Enhance one image and return one new RGBA Pillow image; does not save."""
    return _enhance_one(image, load_config(config_path))


def main(config_path=DEFAULT_CONFIG):
    """Optional folder workflow. All settings come from YAML, never CLI args."""
    config = load_config(config_path)
    if not hasattr(config, "input") or not hasattr(config, "output"):
        raise ValueError("The folder workflow requires 'input' and 'output' in YAML.")
    source = config.input.resolve(strict=True)
    target = config.output.resolve()
    if target == (source if source.is_dir() else source.parent):
        raise ValueError("Choose an output directory different from the input directory.")
    supported = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = ([source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.is_file() and p.suffix.lower() in supported
    ))
    if not files:
        raise ValueError("No supported images found.")
    target.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, 1):
        result = _enhance_one(path, config)
        # Include original extension to avoid collisions such as a.jpg / a.png.
        destination = target / f"{path.name}_enhanced.png"
        suffix = 1
        while True:
            try:
                stream = destination.open("xb")  # Never overwrite existing files.
                break
            except FileExistsError:
                destination = target / f"{path.name}_enhanced_{suffix}.png"
                suffix += 1
        try:
            with stream:
                result.save(stream, format="PNG", icc_profile=result.info["icc_profile"])
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        print(f"[{index}/{len(files)}] Saved {destination}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)

# Upstream interface attribution and license:
# MIT License
# 
# Copyright (c) 2026 Merserk
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
