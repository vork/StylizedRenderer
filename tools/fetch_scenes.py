"""Download Khronos sample assets with their upstream attribution files."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, unquote
from urllib.request import urlopen

ROOT = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models'


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): return
    print(f'Download {path}', flush=True)
    with urlopen(url, timeout=120) as response:
        data = response.read()
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_bytes(data); temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, default=Path('assets'))
    a = p.parse_args()
    manifest = []
    for model, kind in [('Sponza','glTF'), ('DamagedHelmet','glTF-Binary'), ('MetalRoughSpheres','glTF-Binary')]:
        root = a.directory/model
        fetch(f'{ROOT}/{model}/README.md', root/'README.upstream.md')
        suffix = 'gltf' if kind == 'glTF' else 'glb'
        file = root/kind/f'{model}.{suffix}'
        fetch(f'{ROOT}/{model}/{kind}/{model}.{suffix}', file)
        if suffix == 'gltf':
            g = json.loads(file.read_text())
            for item in g.get('buffers', []) + g.get('images', []):
                uri = item.get('uri', '')
                if uri and not uri.startswith('data:'):
                    fetch(f'{ROOT}/{model}/{kind}/{quote(unquote(uri))}', file.parent/unquote(uri))
        manifest.append({'model': model, 'source': f'https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/{model}',
                         'file': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    (a.directory/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
