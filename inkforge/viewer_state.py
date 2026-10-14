"""Transactional scene replacement shared by native drop handling and tests."""
from pathlib import Path
from .scene import load_scene
from .environment import with_ground
from .renderer import Renderer

SCENE_SUFFIXES={'.glb','.gltf','.usd','.usda','.usdc','.usdz'}


def dropped_scene_path(files):
    for filename in files:
        path=Path(filename)
        if path.suffix.lower() in SCENE_SUFFIXES and path.is_file():return path
    raise ValueError('Drop a GLB, glTF, USD, USDA, USDC or USDZ scene file.')


def replace_renderer(previous,raw_scene,ground_mode):
    # Build completely before swapping. A failed import/upload keeps the old scene.
    previous.device.wait()
    scene=with_ground(raw_scene,ground_mode)
    return Renderer(scene,previous.width,previous.height,previous.backend,
                    previous.max_texture,previous.shadow_size,device=previous.device)


def load_dropped_scene(previous,files,ground_mode):
    raw=load_scene(dropped_scene_path(files))
    return replace_renderer(previous,raw,ground_mode),raw
