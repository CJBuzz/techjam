#!/usr/bin/env python3
"""Extract raw-image FFT scores with exact VAE input identity checks."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import PIL, torch, yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "sd15_vae_fft"
SHARED = ROOT / "sb15_fft"
sys.path.insert(0, str(SHARED))
from sb15_fft.perturbations import ExactSeverityTransform, derived_seed, severity_key
from sb15_fft.preprocessing import canonical, pixel_hash
from sb15_fft.spectral_metrics import image_metrics

MANIFEST = SHARED / "results/manifests/frozen_extraction.parquet"
CONFIG = PROJECT / "configs/raw_fft_experiment.yaml"
CONDITIONS = {"clean": ("clean",0.0), "jpeg_q30": ("jpeg",30.0), "blur_2": ("blur",2.0), "resize_0.25x": ("resize",0.25)}
EXPECTED = {"manifest":"097bb22382f950afdb314d894c8ddd0e69e22b6f73bcf3f80006485db58b16e4"}

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20), b""): h.update(b)
    return h.hexdigest()

def atomic_json(obj, path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)

def atomic_parquet(frame, path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp")
    frame.to_parquet(tmp,index=False); check=pd.read_parquet(tmp)
    if len(check)!=len(frame) or check.ordinal.tolist()!=frame.ordinal.tolist(): raise RuntimeError("checkpoint round-trip failed")
    os.replace(tmp,path)

def resolve_ref(registry_path):
    reg=json.loads(registry_path.read_text()); out={}
    for name,item in reg["files"].items():
        p=ROOT/item["path"]
        if digest(p)!=item["sha256"]: raise RuntimeError(f"reference hash mismatch: {name}")
        out[name]=(p,item)
    return reg,out

def config_identity(args, registry, refs, fallback):
    files={"runner":Path(__file__),"perturbations":SHARED/"sb15_fft/perturbations.py","preprocessing":SHARED/"sb15_fft/preprocessing.py","spectral":SHARED/"sb15_fft/spectral_metrics.py","manifest":MANIFEST,"lock":SHARED/"uv.lock","registry":args.reference_registry}
    payload={"config":yaml.safe_load(CONFIG.read_text()),"batch_size":args.batch_size,"backend":"cpu","fallback_tier":fallback,"allow_approximate_inputs":args.allow_approximate_inputs,
             "hashes":{k:digest(v) for k,v in files.items()},"vae_hashes":{k:digest(v[0]) for k,v in refs.items()},
             "versions":{"python":platform.python_version(),"torch":torch.__version__,"pillow":PIL.__version__,"numpy":np.__version__,"pandas":pd.__version__}}
    raw=json.dumps(payload,sort_keys=True,separators=(",",":")).encode(); return hashlib.sha256(raw).hexdigest(),payload

def validate_inventory(manifest):
    required={"ordinal","image_id","relative_path","source_path","severity_seed_key","source_bytes","source_sha256"}
    if not required<=set(manifest): raise RuntimeError(f"manifest missing {required-set(manifest)}")
    if len(manifest)!=11841 or manifest.ordinal.tolist()!=list(range(11841)) or not manifest.image_id.is_unique: raise RuntimeError("invalid frozen manifest")

def validate_reference(frame,name):
    score="fft_high_bandmean_ratio"
    if len(frame)!=11841 or frame.ordinal.tolist()!=list(range(11841)) or not frame.image_id.is_unique: raise RuntimeError(f"invalid reference inventory: {name}")
    if frame.condition.astype(str).nunique()!=1 or str(frame.condition.iloc[0])!=name: raise RuntimeError(f"reference condition mismatch: {name}")
    if not np.isfinite(frame[score]).all(): raise RuntimeError(f"non-finite VAE scores: {name}")

def process_condition(args,name,spec,manifest,reference,outdir,confighash):
    operation,value=spec; out=outdir/f"{name}.parquet"; summary=out.with_suffix(".json")
    prior=pd.DataFrame()
    if args.resume and out.exists():
        prior=pd.read_parquet(out)
        if len(prior) and (prior.configuration_hash.ne(confighash).any() or prior.ordinal.tolist()!=manifest.ordinal.iloc[:len(prior)].tolist() or prior.image_id.tolist()!=manifest.image_id.iloc[:len(prior)].tolist()): raise RuntimeError(f"invalid resume checkpoint {out}")
        if len(prior)==len(manifest): return json.loads(summary.read_text())
    rows=prior.to_dict("records"); started=time.monotonic()
    for pos in range(len(rows),len(manifest)):
        row=manifest.iloc[pos]; ref=reference.iloc[pos]
        if int(row.ordinal)!=int(ref.ordinal) or row.image_id!=ref.image_id: raise RuntimeError(f"row alignment mismatch at {pos}")
        source=Path(row.source_path)
        if not source.exists(): raise FileNotFoundError(source)
        if source.stat().st_size!=int(row.source_bytes) or digest(source)!=row.source_sha256: raise RuntimeError(f"source identity mismatch: {row.image_id}")
        with Image.open(source) as im: image=ImageOps.exif_transpose(im).convert("RGB")
        transformed=ExactSeverityTransform(operation,value,42,row.severity_seed_key)(image)
        th=pixel_hash(np.asarray(transformed,dtype=np.uint8)); tensor,array=canonical(transformed); ch=pixel_hash(array)
        seed=derived_seed(42,row.severity_seed_key,operation,value)
        seed_ok=int(seed)==int(ref.derived_seed); t_ok=th==ref.transformed_sha256; c_ok=ch==ref.canonical_sha256
        exact=seed_ok and t_ok and c_ok
        if not exact and not args.allow_approximate_inputs: raise RuntimeError(f"input identity mismatch {name} ordinal={pos}: seed={seed_ok} transformed={t_ok} canonical={c_ok}")
        met=image_metrics(tensor.unsqueeze(0),torch.zeros_like(tensor).unsqueeze(0))
        rows.append({"ordinal":int(row.ordinal),"image_id":row.image_id,"condition":name,"operation":operation,"value":float(value),"global_seed":42,"derived_seed":int(seed),"severity_key":severity_key(operation,value),"source_sha256":row.source_sha256,"transformed_sha256":th,"canonical_sha256":ch,"reference_transformed_sha256":ref.transformed_sha256,"reference_canonical_sha256":ref.canonical_sha256,"input_identity_status":"exact" if exact else "mismatch","fallback_tier":"E0" if exact else "A1","runtime_seconds":time.monotonic()-started,"configuration_hash":confighash,"method":"raw_image_fft_high_bandmean_ratio",**{"raw_"+k:float(v[0]) for k,v in met.items() if k.startswith("fft_")}})
        if len(rows)%100==0: atomic_parquet(pd.DataFrame(rows),out)
    frame=pd.DataFrame(rows); atomic_parquet(frame,out)
    mismatches=int(frame.input_identity_status.ne("exact").sum())
    result={"status":"complete","condition":name,"rows":len(frame),"unique_image_ids":int(frame.image_id.nunique()),"configuration_hash":confighash,"fallback_tier":"E0" if mismatches==0 else "A1","exact_input_identity":mismatches==0,"input_mismatch_count":mismatches,"score_sha256":digest(out),"path":str(out.relative_to(ROOT))}
    atomic_json(result,summary); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",required=True,choices=("smoke","production")); ap.add_argument("--only",action="append",choices=tuple(CONDITIONS)); ap.add_argument("--resume",action="store_true"); ap.add_argument("--batch-size",type=int,default=16); ap.add_argument("--reference-registry",type=Path,default=PROJECT/"results/selected_evaluation_registry.json"); ap.add_argument("--allow-approximate-inputs",action="store_true"); args=ap.parse_args()
    if digest(MANIFEST)!=EXPECTED["manifest"]: raise RuntimeError("manifest hash mismatch")
    manifest=pd.read_parquet(MANIFEST).sort_values("ordinal").reset_index(drop=True); validate_inventory(manifest)
    registry,refs=resolve_ref(args.reference_registry); frames={k:pd.read_parquet(v[0]).sort_values("ordinal").reset_index(drop=True) for k,v in refs.items()}
    for k,f in frames.items(): validate_reference(f,k)
    if args.mode=="smoke":
        ix=np.linspace(0,len(manifest)-1,16,dtype=int); manifest=manifest.iloc[ix].reset_index(drop=True); frames={k:f.iloc[ix].reset_index(drop=True) for k,f in frames.items()}
    tier="A1" if args.allow_approximate_inputs else "E0"; confighash,identity=config_identity(args,registry,refs,tier)
    outdir=PROJECT/f"results/raw_fft/scores/{args.mode}/{confighash}"; selected=args.only or list(CONDITIONS); results=[]
    for name in selected: results.append(process_condition(args,name,CONDITIONS[name],manifest,frames[name],outdir,confighash))
    report={"status":"complete","mode":args.mode,"configuration_hash":confighash,"configuration_identity":identity,"conditions":results,"divergence_ledger":[]}
    atomic_json(report,PROJECT/f"results/raw_fft/logs/{args.mode}_report.json"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()

