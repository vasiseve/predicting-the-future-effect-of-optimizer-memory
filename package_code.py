
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def release_files(root):
    excluded={'__pycache__','.git','.venv','results','.ipynb_checkpoints'}
    return sorted(p for p in root.rglob('*') if p.is_file() and not any(part in excluded for part in p.relative_to(root).parts) and p.name not in {'.DS_Store','MANIFEST.sha256'} and p.suffix not in {'.pyc','.zip','.pt','.pth'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    files=release_files(root)
    forbidden=['/'+'Users/','/'+'home/','vas'+'iseve','Post'+'Doc']
    for p in files:
        if p.suffix in {'.py','.ipynb','.md','.csv','.json','.txt'}:
            data=p.read_text()
            for marker in forbidden:
                if marker in data:raise SystemExit(f'Identifying path or text in {p.relative_to(root)}')
        if p.suffix=='.ipynb':
            nb=json.loads(p.read_text())
            for cell in nb['cells']:
                if cell['cell_type']=='code' and (cell.get('outputs') or cell.get('execution_count') is not None):
                    raise SystemExit(f'Notebook execution metadata in {p.relative_to(root)}')
    manifest=root/'MANIFEST.sha256'
    manifest.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n' for p in files))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(files+[manifest]):
            info=zipfile.ZipInfo(root.name+'/'+p.relative_to(root).as_posix(),date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes())
    with zipfile.ZipFile(args.output) as z:
        if z.testzip() is not None:raise SystemExit('Archive integrity check failed')
    checksum=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.zip.sha256').write_text(checksum+'  '+args.output.name+'\n')
    print(json.dumps({'archive':args.output.name,'files':len(files)+1,'bytes':args.output.stat().st_size,'sha256':checksum},indent=2))


if __name__=='__main__':main()
