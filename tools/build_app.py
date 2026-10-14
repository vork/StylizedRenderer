"""Build a self-contained app for the current OS/architecture (not a cross build)."""
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--clean',
        '--distpath',str(root/'dist'),'--workpath',str(root/'build'/'app'),
        str(root/'packaging'/'Inkforge.spec')],cwd=root,check=True)
    system=platform.system();dist=root/'dist'
    archive=dist/f'Inkforge-{system}-{platform.machine()}.zip'
    if system=='Darwin':
        subprocess.run(['ditto','-c','-k','--sequesterRsrc','--keepParent',str(dist/'Inkforge.app'),str(archive)],check=True)
    else:shutil.make_archive(str(archive.with_suffix('')),'zip',root_dir=dist,base_dir='Inkforge')
    print(f'Shareable archive: {archive}')


if __name__=='__main__':main()
