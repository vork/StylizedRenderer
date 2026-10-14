"""Replay a viewer capture's exact camera/settings for shader diagnostics."""
import argparse
import json
from pathlib import Path
from inkforge.renderer import Renderer,Style
from inkforge.scene import load_scene


def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--name',default='replay')
    p.add_argument('--hour',type=float);p.add_argument('--frequency',type=float);a=p.parse_args()
    data=json.loads(a.capture.read_text());style=Style(**data['style'])
    if a.hour is not None:style.time_of_day=a.hour
    if a.frequency is not None:style.dot_frequency=a.frequency
    r=Renderer(load_scene(data['source']),*data['render_resolution'])
    out=Path('output/closeup-review');out.mkdir(parents=True,exist_ok=True)
    for name,debug in [('beauty',0),('printed',11),('cast',9),('cascades',8)]:
        r.save(out/f'{a.name}-{name}.png',data['eye'],data['target'],style,data['fov'],debug,time_seconds=data['cloud_time_seconds'])
    style.penumbra_strength=0
    r.save(out/f'{a.name}-no-stipple.png',data['eye'],data['target'],style,data['fov'],time_seconds=data['cloud_time_seconds'])


if __name__=='__main__':main()
