from pathlib import Path
import os
import sys
from PyInstaller.utils.hooks import collect_data_files,collect_dynamic_libs,collect_submodules,copy_metadata

root=Path(SPECPATH).parent
datas=collect_data_files('inkforge')
datas+=collect_data_files('slangpy',excludes=['tests/**','benchmarks/**','include/**','**/*.a','**/*.pyi'])
datas+=collect_data_files('pxr',excludes=['**/testenv/**','**/*.pyi'])
for package in ('slangpy','usd-core','pygltflib','Pillow','dataclasses-json','typing-inspect','typing-extensions','PyInstaller'):
    datas+=copy_metadata(package)
datas.append((str(root/'docs'/'sharing.md'),'.'))
binaries=collect_dynamic_libs('slangpy')+collect_dynamic_libs('pxr')
hidden=collect_submodules('pxr',filter=lambda name:'.testenv' not in name)
hidden+=collect_submodules('slangpy',filter=lambda name:'.tests' not in name and '.benchmarks' not in name)
a=Analysis([str(root/'inkforge'/'desktop.py')],pathex=[str(root)],binaries=binaries,datas=datas,
    hiddenimports=hidden,excludes=['torch','pytest','IPython','matplotlib'],
    runtime_hooks=[str(root/'packaging'/'runtime.py')])
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='Inkforge',debug=False,strip=False,upx=False,
    console=False,argv_emulation=sys.platform=='darwin',
    codesign_identity=os.environ.get('INKFORGE_CODESIGN_IDENTITY'))
collection=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='Inkforge')
if sys.platform=='darwin':
    app=BUNDLE(collection,name='Inkforge.app',bundle_identifier='dev.inkforge.viewer',
        info_plist={'CFBundleShortVersionString':'0.1.0','NSHighResolutionCapable':True,
                    'CFBundleDocumentTypes':[{'CFBundleTypeName':'3D Scene','CFBundleTypeRole':'Viewer',
                    'CFBundleTypeExtensions':['glb','gltf','usd','usda','usdc','usdz']} ]})
