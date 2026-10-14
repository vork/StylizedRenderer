"""Linear-light mip generation; color textures use hardware sRGB sampling."""
import numpy as np
from PIL import Image


def mip_chain(image, max_size, srgb):
    pixels=np.asarray(image.convert('RGBA'),dtype=np.float32)/255
    if srgb:
        rgb=pixels[:,:,:3]
        pixels[:,:,:3]=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    scale=min(1,max_size/max(image.size))
    w,h=max(1,round(image.width*scale)),max(1,round(image.height*scale))
    levels=[]
    while True:
        if pixels.shape[:2]!=(h,w):
            pixels=np.stack([np.asarray(Image.fromarray(pixels[:,:,c]).resize((w,h),Image.Resampling.BOX)) for c in range(4)],axis=-1)
        encoded=pixels.copy()
        if srgb:
            rgb=encoded[:,:,:3]
            encoded[:,:,:3]=np.where(rgb<=.0031308,rgb*12.92,1.055*np.maximum(rgb,0)**(1/2.4)-.055)
        levels.append(np.uint8(np.clip(encoded,0,1)*255+.5))
        if w==1 and h==1:break
        w,h=max(1,w//2),max(1,h//2)
    return levels
