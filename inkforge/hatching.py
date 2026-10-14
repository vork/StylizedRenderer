"""Seamless, deterministic pencil stroke tonal maps; RGBA = 0/90/+45/-45°."""
import numpy as np
from PIL import Image,ImageDraw
from pathlib import Path


def create_textures(size=512):
    y,x=np.mgrid[:size,:size].astype(np.float32)/size
    directions=[(x,y),(y,x),(x,y-x),(x,y+x)]
    levels=[np.zeros((size,size,4),np.float32) for _ in range(4)]
    for channel,(along,across) in enumerate(directions):
        for index in range(16):
            phase=index*2.399963
            bend=.0016*np.sin(2*np.pi*along*3+phase)+.0007*np.sin(2*np.pi*along*7+phase)
            distance=np.abs((across-index/16-bend+.5)%1-.5)
            width=.0036*(.8+.2*np.sin(2*np.pi*along*5+phase))
            ink=np.clip((width+1/size-distance)*size,0,1)
            ink*=.82+.18*np.sin(2*np.pi*along*11+phase)**2
            for level,count in enumerate([2,4,8,16]):
                if index%(16//count)==0:
                    levels[level][:,:,channel]=np.maximum(levels[level][:,:,channel],ink)
    return [np.uint8(np.clip(level*255+.5,0,255)) for level in levels]


def main():
    root=Path(__file__).parent/'resources';root.mkdir(exist_ok=True)
    textures=create_textures()
    for index,data in enumerate(textures):Image.fromarray(data).save(root/f'hatching-{index}.png')
    # Contact sheet shows data channels as ink on paper, not RGBA transparency.
    preview=Image.new('RGB',(1040,1104),(244,240,228));draw=ImageDraw.Draw(preview)
    for row,data in enumerate(textures):
        for col,label in enumerate(['Horizontal','Vertical','Diagonal +45','Diagonal -45']):
            ink=Image.fromarray(255-data[:,:,col]).resize((256,256),Image.Resampling.LANCZOS).convert('RGB')
            preview.paste(ink,(col*260,row*276+20))
            draw.text((col*260+4,row*276+4),f'{label} / {2**(row+1)} strokes',fill=(30,30,30))
    out=Path('output');out.mkdir(exist_ok=True);preview.save(out/'hatching-texture-sheet.png')


if __name__=='__main__':main()
