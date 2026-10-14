"""Real GPU captures of sky, celestial objects and the automatic object ground."""
from pathlib import Path
import numpy as np
from PIL import Image
from inkforge.scene import load_scene
from inkforge.environment import with_ground
from inkforge.renderer import Renderer,Style


def main():
    scene=load_scene('assets/DamagedHelmet/glTF-Binary/DamagedHelmet.glb')
    r=Renderer(with_ground(scene),960,600)
    s=Style();out=Path('output')
    for name,hour in [('day',10),('sunset',17.6),('night',22)]:
        s.time_of_day=hour
        r.save(out/f'sky-{name}-object.png',[2,.7,4],[0,.35,0],s)
        # Skyward view follows the visible sun/moon, leaving ample surrounding sky.
        from inkforge.environment import lighting_state
        celestial=lighting_state(scene,s)['sun_direction']
        if celestial[1]<0:celestial=-celestial
        eye=np.array([0.,3,0]);target=eye+celestial*5
        r.save(out/f'sky-{name}.png',eye,target,s)
    s.time_of_day=10
    for stamp in (0,30):
        r.save(out/f'sky-clouds-{stamp}.png',[0,3,0],[-2,5,4],s,time_seconds=stamp)
    panels=[Image.open(out/f'sky-{name}.png') for name in ('day','sunset','night')]
    sheet=Image.new('RGB',(960,600*3))
    for i,panel in enumerate(panels):sheet.paste(panel,(0,600*i))
    sheet.save(out/'sky-time-of-day.png')


if __name__=='__main__':main()
