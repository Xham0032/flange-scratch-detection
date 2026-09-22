"""Evaluate the fixed custom split without exporting images or contours."""
from pathlib import Path
import argparse,csv,json
from dataclasses import asdict
import cv2
import numpy as np
import flange
import washer_core as core

def components(mask):
    n,lab,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),connectivity=8)
    ids=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=8);ids=ids[ids!=0]
    filtered=np.isin(lab,ids).astype(np.uint8)
    n,labels=cv2.connectedComponents(filtered,connectivity=8)
    return filtered,labels,n-1

def metrics(pred,truth):
    pm,pl,pn=components(pred);tm,tl,tn=components(truth)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    hp=np.unique(pl[cv2.dilate(tm,kernel)>0]);ht=np.unique(tl[cv2.dilate(pm,kernel)>0])
    p=np.count_nonzero(hp)/pn if pn else 0.
    r=np.count_nonzero(ht)/tn if tn else 0.
    return dict(P=float(p),R=float(r),F1=float(2*p*r/(p+r) if p+r else 0.))

def stages(images,profile):
    _,roi=core.roi_for(images,profile)
    base,union=core.extract(images,roi,profile)
    geometry=[f for f in base if f['length']>=profile.length and profile.solidity_low<=f['solidity']<=profile.solidity_high and f['thickness']<=profile.thickness]
    support=[f for f in geometry if f['light_support']>=profile.support]
    contrast=[f for f in geometry if abs(f['best_contrast'])>=profile.contrast_floor]
    joint=[f for f in support if abs(f['best_contrast'])>=profile.contrast_floor]
    return dict(candidate_union=union,**{name:core.paint(fs,roi.shape) for name,fs in [('geometry',geometry),('directional_support',support),('contrast',contrast),('joint',joint)]})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--split',type=Path,default=Path(__file__).parent/'splits/washer_20_30.csv')
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/washer_parameters.json')
    parser.add_argument('--output',type=Path,required=True,help='Local JSON metrics only')
    args=parser.parse_args()
    specs=list(csv.DictReader(args.split.open(encoding='utf-8-sig')))
    if len(specs)!=50 or sum(s['partition']=='tuning' for s in specs)!=20 or sum(s['partition']=='heldout' for s in specs)!=30 or len({(s['source_split'],s['sample_id']) for s in specs})!=50:
        raise ValueError('Expected 50 unique physical samples in a 20/30 split')
    profile=core.Profile(**json.loads(args.config.read_text(encoding='utf8')))
    rows=[]
    for spec in specs:
        # Only the explicit safe dataset identifiers determine local input paths.
        sid=spec['sample_id'];split=spec['source_split']
        if split not in ('Train','Test') or not sid.isdigit():raise ValueError('Invalid sample identifier')
        folder=args.data_root/split/sid
        ims=[flange.read_gray(folder/f'washer_{sid}_{k}.png') for k in (101,104,107,110)]
        truth=flange.read_gray(folder/f'washer_{sid}_mask.png')
        for name,pred in stages(ims,profile).items():rows.append(dict(**spec,stage=name,**metrics(pred,truth)))
        print(spec['partition'],split,sid,flush=True)
    summary=[]
    for partition in ['tuning','heldout']:
        for name in ['candidate_union','geometry','directional_support','contrast','joint']:
            ss=[r for r in rows if r['partition']==partition and r['stage']==name]
            summary.append(dict(partition=partition,stage=name,n=len(ss),**{k:float(np.mean([r[k] for r in ss])) for k in ['P','R','F1']}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(profile=asdict(profile),opencv=cv2.__version__,summary=summary,per_sample=rows),indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
