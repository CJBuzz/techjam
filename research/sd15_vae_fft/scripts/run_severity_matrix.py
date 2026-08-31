#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[2]; PROJECT=ROOT/"sd15_vae_fft"; SHARED=ROOT/"sb15_fft"
sys.path.insert(0,str(SHARED))
from sb15_fft.perturbations import SEVERITY_SPECS,ExactSeverityTransform,derived_seed,severity_key
from sb15_fft.preprocessing import canonical,pixel_hash
from sb15_fft.spectral_metrics import image_metrics

MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"; REVISION="451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
def atomic_parquet(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp"); frame.to_parquet(tmp,index=False); os.replace(tmp,path)
def atomic_json(value,path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
def experiment_hash(batch_size):
    value={"config":yaml.safe_load((PROJECT/"configs/experiment.yaml").read_text()),"batch_size":batch_size,
      "manifest_sha256":digest(SHARED/"results/manifests/frozen_extraction.parquet"),"lock_sha256":digest(SHARED/"uv.lock"),
      "code":{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),SHARED/"sb15_fft/perturbations.py",SHARED/"sb15_fft/preprocessing.py",SHARED/"sb15_fft/spectral_metrics.py"]}}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
class VAEReconstructor:
    def __init__(self):
        from diffusers import AutoencoderKL
        self.vae=AutoencoderKL.from_pretrained(MODEL,subfolder="vae",revision=REVISION,torch_dtype=torch.float16,local_files_only=True).eval().to("cuda")
        self.vae.requires_grad_(False)
    def __call__(self,x):
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
            # Standalone VAE round trip: scaling_factor is not applied because the latent never enters the UNet.
            z=self.vae.encode(x.to("cuda",dtype=torch.float16)*2-1).latent_dist.mode()
            y=(self.vae.decode(z).sample.float().clamp(-1,1)+1)/2
        if not torch.isfinite(y).all(): raise RuntimeError("non-finite VAE reconstruction")
        return y.cpu()
def load_image(path):
    with Image.open(path) as source: return ImageOps.exif_transpose(source).convert("RGB")
def save_png(y,path):
    a=(y.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp"); Image.fromarray(a,"RGB").save(tmp,format="PNG")
    with Image.open(tmp) as check: check.load(); assert check.mode=="RGB" and check.size==(512,512)
    os.replace(tmp,path); return digest(path),path.stat().st_size
def validate_existing(frame,records,cfg):
    if len(frame)>len(records) or frame.configuration_hash.nunique()!=1 or frame.configuration_hash.iloc[0]!=cfg: raise RuntimeError("resume configuration mismatch")
    expected=[int(x.ordinal) for x in records[:len(frame)]]
    if frame.ordinal.tolist()!=expected or not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all(): raise RuntimeError("invalid resume prefix")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("smoke","pilot","production"),required=True); ap.add_argument("--resume",action="store_true"); ap.add_argument("--only",action="append",default=[],help="Run only this exact condition key; repeat as needed"); a=ap.parse_args()
    manifest=pd.read_parquet(SHARED/"results/manifests/frozen_extraction.parquet")
    selected=manifest.iloc[np.linspace(0,len(manifest)-1,16,dtype=int)] if a.mode=="smoke" else manifest.iloc[:200] if a.mode=="pilot" else manifest
    records=list(selected.itertuples(index=False)); batch_size=4; cfg=experiment_hash(batch_size); print(f"configuration_hash={cfg}",flush=True)
    known={severity_key(*spec) for spec in SEVERITY_SPECS}; unknown=set(a.only)-known
    if unknown: raise ValueError(f"unknown conditions: {sorted(unknown)}")
    selected_specs=[spec for spec in SEVERITY_SPECS if not a.only or severity_key(*spec) in a.only]
    model=VAEReconstructor(); torch.cuda.reset_peak_memory_stats(); summaries=[]; run_start=time.perf_counter()
    for operation,value in selected_specs:
        key=severity_key(operation,value); out=PROJECT/"results/scores"/a.mode/cfg/f"{key}.parquet"; rows=[]
        if a.resume and out.is_file(): rows=pd.read_parquet(out).sort_values("ordinal").to_dict("records"); validate_existing(pd.DataFrame(rows),records,cfg)
        if len(rows)==len(records): print(f"{key}: validated complete",flush=True); continue
        start=time.perf_counter(); retries=sum(int(x["retry_count"]) for x in rows)
        for offset in range(len(rows),len(records),batch_size):
            batch=records[offset:offset+batch_size]; xs=[]; meta=[]
            for r in batch:
                im=load_image(Path(r.source_path)); transformed=ExactSeverityTransform(operation,value,42,r.severity_seed_key)(im)
                ta=np.asarray(transformed,dtype=np.uint8); x,ca=canonical(transformed); xs.append(x); meta.append((r,pixel_hash(ta),pixel_hash(ca),derived_seed(42,r.severity_seed_key,operation,value)))
            tick=time.perf_counter()
            for attempt in range(3):
                try: y=model(torch.stack(xs)); break
                except Exception:
                    retries+=1; torch.cuda.empty_cache()
                    if attempt==2: raise
            runtime=(time.perf_counter()-tick)/len(batch); metrics=image_metrics(torch.stack(xs),y)
            for i,(r,th,ch,seed) in enumerate(meta):
                target=PROJECT/"artifacts"/a.mode/cfg/key/f"{r.ordinal:05d}_{r.image_id}.png"; rh,rb=save_png(y[i],target)
                row={"ordinal":int(r.ordinal),"image_id":r.image_id,"condition":key,"operation":operation,"value":value,"global_seed":42,"derived_seed":seed,"severity_key":r.severity_seed_key,
                  "source_sha256":r.source_sha256,"transformed_sha256":th,"canonical_sha256":ch,"reconstruction_sha256":rh,"reconstruction_path":str(target.relative_to(PROJECT)),
                  "reconstruction_bytes":rb,"runtime_seconds":runtime,"retry_count":retries,"configuration_hash":cfg,"method":"sd15_vae_posterior_mode"}
                row.update({k:float(v[i]) for k,v in metrics.items()}); rows.append(row)
            done=offset+len(batch)
            if done%100==0 or done==len(records):
                atomic_parquet(pd.DataFrame(rows).sort_values("ordinal"),out); elapsed=time.perf_counter()-start
                print(f"{key} {done}/{len(records)} elapsed={elapsed:.1f}s throughput={(done-len(rows)+len(rows))/max(elapsed,1e-9):.2f}/s retries={retries} current_vram={torch.cuda.memory_reserved()} peak_vram={torch.cuda.max_memory_reserved()}",flush=True)
        s={"condition":key,"rows":len(rows),"elapsed_seconds":time.perf_counter()-start,"retries":retries,"score_sha256":digest(out),"reconstruction_bytes":sum(x["reconstruction_bytes"] for x in rows)}; atomic_json(s,out.with_suffix(".json")); summaries.append(s)
    report={"mode":a.mode,"method":"sd15_vae_posterior_mode","configuration_hash":cfg,"rows_per_condition":len(records),"conditions":[severity_key(*spec) for spec in selected_specs],"batch_size":batch_size,
      "elapsed_seconds":time.perf_counter()-run_start,"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"summaries":summaries}
    atomic_json(report,PROJECT/"results/logs"/f"{a.mode}_report.json"); print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__": main()
