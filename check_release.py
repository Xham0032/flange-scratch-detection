"""Validate the reviewed text-only manifest and optionally build an upload ZIP."""
from pathlib import Path
import argparse,csv,hashlib,json,re,zipfile

FILES={'.gitignore','README.md','requirements.txt','flange.py','washer_core.py',
       'evaluate_washer.py','check_release.py','configs/flange_parameters.json',
       'configs/washer_parameters.json','splits/washer_20_30.csv'}

def validate(root):
    found=set()
    for p in root.rglob('*'):
        rel=p.relative_to(root)
        if '.git' in rel.parts or '__pycache__' in rel.parts:continue
        if p.is_symlink():raise ValueError(f'Symlink prohibited: {rel}')
        if p.is_file():found.add(rel.as_posix())
    if found!=FILES:raise ValueError(f'Unexpected/missing files: {sorted(found^FILES)}')
    manifest={}
    for name in sorted(FILES):
        raw=(root/name).read_bytes();s=raw.decode('utf8')
        if b'\x00' in raw:raise ValueError(f'Binary content: {name}')
        # Check content, not just extension; forbid embedded images and long binary encodings.
        if re.search('data:'+'image/|'+r'[A-Za-z0-9+/]{1024,}',s):raise ValueError(f'Possible encoded image: {name}')
        if re.search(r'[A-Z]:[/\\]|'+ '/Us'+'ers/|/ho'+'me/',s):raise ValueError(f'Local absolute path: {name}')
        if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}',s):raise ValueError(f'Possible credential: {name}')
        if name.endswith('.py'):compile(s,name,'exec')
        manifest[name]=hashlib.sha256(raw).hexdigest()
    with (root/'splits/washer_20_30.csv').open(encoding='utf8',newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['partition','source_split','sample_id']
        rows=list(reader)
    assert len(rows)==50 and len({(r['source_split'],r['sample_id']) for r in rows})==50
    assert sum(r['partition']=='tuning' for r in rows)==20
    assert sum(r['partition']=='heldout' for r in rows)==30
    assert all(r['source_split'] in ['Train','Test'] and re.fullmatch(r'\d{3}',r['sample_id']) for r in rows)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--zip',type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parent
    manifest=validate(root)
    if args.zip:
        dest=args.zip.resolve()
        if dest.is_relative_to(root):raise ValueError('Write ZIP outside the source directory')
        dest.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            for name in sorted(FILES):z.write(root/name,'flange-scratch-reproducibility/'+name)
        with zipfile.ZipFile(dest) as z:
            assert set(z.namelist())=={'flange-scratch-reproducibility/'+n for n in FILES}
            assert all(hashlib.sha256(z.read('flange-scratch-reproducibility/'+n)).hexdigest()==h for n,h in manifest.items())
    print(json.dumps({'status':'PASS','text_files':len(manifest),'files':manifest},indent=2))
