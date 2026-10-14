"""Verify a relocated bundle without using the development Python environment."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'output'/'bundle-validation';output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='inkforge-relocation-') as directory:
        folder=Path(directory);bundle=folder/'Inkforge.app'
        shutil.copytree(root/'dist'/'Inkforge.app',bundle,symlinks=True)
        subprocess.run(['codesign','--verify','--deep','--strict',str(bundle)],check=True)
        # No non-system absolute dependencies should escape the bundle.
        inspected=0
        for path in (bundle/'Contents'/'Frameworks').rglob('*'):
            if path.is_symlink() or not path.is_file():continue
            with path.open('rb') as stream:magic=stream.read(4)
            if magic not in (b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xca\xfe\xba\xbe'):continue
            inspected+=1
            linked=subprocess.check_output(['otool','-L',str(path)],text=True)
            for line in linked.splitlines()[1:]:
                dependency=line.strip().split(' (')[0]
                assert not dependency.startswith('/') or dependency.startswith(('/usr/lib/','/System/Library/')),dependency
        env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','PXR_PLUGINPATH_NAME')}
        exe=bundle/'Contents'/'MacOS'/'Inkforge'
        scenes=[root/'assets'/'material-showroom.usda',root/'assets'/'DamagedHelmet'/'glTF-Binary'/'DamagedHelmet.glb']
        for scene in scenes:
            local=folder/scene.name;shutil.copy2(scene,local)
            destination=output/(scene.stem+'.png')
            subprocess.run([str(exe),str(local),'--output',str(destination),'--width','640','--height','400'],
                           cwd=folder,env=env,check=True,timeout=120)
            assert destination.is_file()
        report={'relocated':True,'native_libraries_checked':inspected,'formats':['USD','GLB'],'signature':'ad-hoc, verified','notarized':False}
        (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
