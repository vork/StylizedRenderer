"""Make OpenUSD's plugin discovery independent of the launch directory."""
import os
import sys
from pathlib import Path

root=Path(sys._MEIPASS)
os.environ['PXR_PLUGINPATH_NAME']=str(root/'pxr'/'pluginfo')+os.pathsep+os.environ.get('PXR_PLUGINPATH_NAME','')
