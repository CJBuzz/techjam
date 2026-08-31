#!/usr/bin/env python3
"""Freeze raw scores, then perform paired raw-versus-VAE evaluation."""
from __future__ import annotations
import argparse,hashlib,json,os,platform
from pathlib import Path
import numpy as np,pandas as pd,sklearn
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[2]; PROJECT=ROOT/"sd15_vae_fft"; LABELS=ROOT/"data/manifests/wildfake_test_labels.parquet"
LABEL_HASH="0e0ec3902ce3f07e08b3651c8972af71fae07447f9049e62c7c7176e9d525c7f"; RAW="raw_fft_high_bandmean_ratio"; VAE="fft_high_bandmean_ratio"
EXPECTED={"clean":0.9994416117,"jpeg_q30":0.9999971298,"blur_2":0.9311046094,"resize_0.25x":0.9951183357}

def digest(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def atomic(v,p):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+".tmp"); t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n"); os.replace(t,p)
def threshold(y,s):
 u=np.unique(s); c=np.r_[-np.inf,(u[:-1]+u[1:])/2,np.inf]; med=np.median(s)
 return float(max(c,key=lambda t:(balanced_accuracy_score(y,s>=t),-abs(t-med),-t)))
def metrics(y,s,t):
 pred=s>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {"auroc":float(roc_auc_score(y,s)),"average_precision":float(average_precision_score(y,s)),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall_tpr":float(tp/(tp+fn)),"specificity":float(tn/(tn+fp)),"fpr":float(fp/(fp+tn)),"f1":float(f1_score(y,pred,zero_division=0)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
def ci(a): return [float(x) for x in np.percentile(a,[2.5,97.5])]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--allow-label-read",action="store_true"); a=ap.parse_args()
 if not a.allow_label_read: raise RuntimeError("requires --allow-label-read")
 report=json.loads((PROJECT/"results/raw_fft/logs/production_report.json").read_text()); cfg=report["configuration_hash"]
 rawfiles={x["condition"]:ROOT/x["path"] for x in report["conditions"]}; names=("clean","jpeg_q30","blur_2","resize_0.25x")
 if set(rawfiles)!=set(names): raise RuntimeError("production report must contain all four conditions")
 vaereg_path=PROJECT/"results/selected_evaluation_registry.json"; vaereg=json.loads(vaereg_path.read_text()); rawframes={}; vaeframes={}; ids=None
 for n in names:
  r=pd.read_parquet(rawfiles[n]); item=vaereg["files"][n]; vp=ROOT/item["path"]
  if digest(vp)!=item["sha256"]: raise RuntimeError(f"VAE hash mismatch {n}")
  v=pd.read_parquet(vp); 
  if len(r)!=11841 or r.input_identity_status.ne("exact").any() or r.configuration_hash.ne(cfg).any(): raise RuntimeError(f"invalid raw frame {n}")
  if ids is None: ids=r.image_id.tolist()
  if r.image_id.tolist()!=ids or v.image_id.tolist()!=ids: raise RuntimeError(f"ID order mismatch {n}")
  rawframes[n]=r; vaeframes[n]=v
 registry={"status":"frozen before label read","fallback_tier":"E0","exact_input_identity":True,"configuration_hash":cfg,"reference_registry_sha256":digest(vaereg_path),"files":{n:{"path":str(rawfiles[n].relative_to(ROOT)),"sha256":digest(rawfiles[n]),"rows":11841,"input_identity_status":"exact"} for n in names}}
 rp=PROJECT/"results/raw_fft/selected_evaluation_registry.json"; atomic(registry,rp)
 if digest(LABELS)!=LABEL_HASH: raise RuntimeError("label hash mismatch")
 labels=pd.read_parquet(LABELS); joined=rawframes["clean"][["image_id"]].merge(labels,on="image_id",validate="one_to_one",sort=False); y=joined.label.to_numpy()
 rs={n:rawframes[n][RAW].to_numpy() for n in names}; vs={n:vaeframes[n][VAE].to_numpy() for n in names}
 for n in names:
  if abs(roc_auc_score(y,vs[n])-EXPECTED[n])>5e-11: raise RuntimeError(f"stored VAE AUROC failed reproduction: {n}")
 rt=threshold(y,rs["clean"]); vt=threshold(y,vs["clean"]); groups=[np.flatnonzero(y==c) for c in (0,1)]; rng=np.random.default_rng(20260830); idx=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(2000)]
 rows=[]
 for n in names:
  rm=metrics(y,rs[n],rt); vm=metrics(y,vs[n],vt); boot=np.asarray([[roc_auc_score(y[i],rs[n][i]),roc_auc_score(y[i],vs[n][i]),average_precision_score(y[i],rs[n][i]),average_precision_score(y[i],vs[n][i]),balanced_accuracy_score(y[i],rs[n][i]>=rt),balanced_accuracy_score(y[i],vs[n][i]>=vt)] for i in idx])
  raw_clean=np.asarray([roc_auc_score(y[i],rs["clean"][i]) for i in idx]); vae_clean=np.asarray([roc_auc_score(y[i],vs["clean"][i]) for i in idx])
  row={"condition":n,"raw":rm,"vae":vm,"raw_auroc_ci95":ci(boot[:,0]),"vae_auroc_ci95":ci(boot[:,1]),"vae_minus_raw_auroc":vm["auroc"]-rm["auroc"],"vae_minus_raw_auroc_ci95":ci(boot[:,1]-boot[:,0]),"raw_change_from_clean":rm["auroc"]-roc_auc_score(y,rs["clean"]),"raw_change_from_clean_ci95":ci(boot[:,0]-raw_clean),"vae_change_from_clean":vm["auroc"]-roc_auc_score(y,vs["clean"]),"vae_change_from_clean_ci95":ci(boot[:,1]-vae_clean),"difference_in_degradation":(vm["auroc"]-roc_auc_score(y,vs["clean"]))-(rm["auroc"]-roc_auc_score(y,rs["clean"])),"difference_in_degradation_ci95":ci((boot[:,1]-vae_clean)-(boot[:,0]-raw_clean)),"raw_ap_ci95":ci(boot[:,2]),"vae_ap_ci95":ci(boot[:,3]),"balanced_accuracy_difference":vm["balanced_accuracy"]-rm["balanced_accuracy"],"balanced_accuracy_difference_ci95":ci(boot[:,5]-boot[:,4])}; rows.append(row)
 out={"status":"complete","fallback_tier":"E0","exact_input_identity":True,"manifest_sha256":report["configuration_identity"]["hashes"]["manifest"],"label_sha256":LABEL_HASH,"reference_registry_sha256":digest(vaereg_path),"raw_registry_sha256":digest(rp),"code_sha256":{"runner":digest(PROJECT/"scripts/run_raw_fft_severity.py"),"evaluator":digest(Path(__file__))},"environment":{"python":platform.python_version(),"sklearn":sklearn.__version__},"conditions":list(names),"n_per_condition":11841,"class_counts":{"real":3998,"fake":7843},"score_name":RAW,"score_direction":"higher is more likely fake","thresholds":{"raw":rt,"vae":vt,"status":"post-hoc clean oracle; fixed across perturbations"},"bootstrap":{"seed":20260830,"replicates":2000,"paired":True,"class_stratified":True},"input_mismatch_counts":{n:0 for n in names},"excluded_rows":{"count":0,"reasons":{}},"divergence_ledger":[],"results":rows}
 op=PROJECT/"results/metrics/raw_fft_vs_vae_selected_production.json"; atomic(out,op)
 flat=pd.json_normalize(rows,sep="_"); flat.to_csv(op.with_suffix(".csv"),index=False)
 # The durable summary lives in README.md; JSON/CSV remain the result source of truth.
 print(flat[["condition","raw_auroc","vae_auroc","vae_minus_raw_auroc","difference_in_degradation"]].to_string(index=False))
if __name__=="__main__": main()
