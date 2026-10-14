"""Render the sphere shadow boundary and its isolated shading stages."""
import argparse
from pathlib import Path
from dataclasses import replace
from PIL import Image,ImageDraw
from inkforge.renderer import Renderer,Style
from inkforge.scene import load_scene


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='before');a=p.parse_args()
    out=Path('output/sphere-review');out.mkdir(parents=True,exist_ok=True)
    r=Renderer(load_scene('assets/material-showroom.usda'),1280,800)
    s=Style(ao_radius=.7);eye=[4,4,-7];target=[0,.5,1]
    stages=[('beauty',s,0),('no-stipple',replace(s,penumbra_strength=0),0),
            ('no-hatch',replace(s,hatching=0),0),('cast',s,9),('terminator',s,10),('printed',s,11)]
    sheet=Image.new('RGB',(1200,900),(24,26,31));d=ImageDraw.Draw(sheet)
    for i,(name,style,debug) in enumerate(stages):
        im=r.save(out/f'{a.name}-{name}.png',eye,target,style,debug=debug)
        crop=im.crop((350,305,650,505)).resize((600,400))
        x=(i%2)*600;y=(i//2)*300
        crop=crop.resize((450,270));sheet.paste(crop,(x+75,y+25))
        d.text((x+15,y+7),name,fill='white')
    sheet.save(out/f'{a.name}-stages.png')


if __name__=='__main__':main()
