"""Reproduce shadow closeups, preserving the preceding render for comparison."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from inkforge.scene import load_scene
from inkforge.renderer import Renderer, Style


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='after')
    parser.add_argument('--compare-only',action='store_true')
    args=parser.parse_args()
    out=Path('output/shadow-review');out.mkdir(parents=True,exist_ok=True)
    if not args.compare_only:
        scene=load_scene(Path.home()/'Downloads/Bistro_Godot.glb')
        renderer=Renderer(scene,1280,800)
        for name,debug in [(args.name,0),(args.name+'-printed',11)]:
            renderer.save(out/(name+'.png'),[-5,2.2,6],[2,4,0],Style(),debug=debug)
    boxes=[(480,310,680,575),(875,475,1200,560)]
    sheet=Image.new('RGB',(1300,760),(24,26,31));draw=ImageDraw.Draw(sheet)
    for column,name in enumerate(['before',args.name]):
        im=Image.open(out/(name+'.png'))
        draw.text((column*650+16,12),'Before' if column==0 else 'Revised shadows / stippling',fill='white')
        sheet.paste(im.crop(boxes[0]).resize((400,530)),(column*650+125,35))
        sheet.paste(im.crop(boxes[1]).resize((650,170)),(column*650,580))
    sheet.save(out/(args.name+'-comparison.png'))


if __name__=='__main__':main()
