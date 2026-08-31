#!/usr/bin/env python3
"""Score the completed clean + selected production cells; sole label reader."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[2]; PROJECT=ROOT/"sd15_vae_fft"; LABELS=ROOT/"data/manifests/wildfake_test_labels.parquet"
LABEL_HASH="0e0ec3902ce3f07e08b3651c8972af71fae07447f9049e62c7c7176e9d525c7f"; SCORE="fft_high_bandmean_ratio"
OLD="9199b8664536bcec9c0d68aa28a98911486cc282f1fdeadbaa639dcd9ac36db4"; NEW="94bed5836585cf21612a250296de72016ee4db90b4920f26e85c1da9dff2ffad"
FILES={"clean":PROJECT/f"results/scores/production/{OLD}/clean.parquet",**{k:PROJECT/f"results/scores/production/{NEW}/{k}.parquet" for k in ("jpeg_q30","blur_2","resize_0.25x")}}
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
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--allow-label-read",action="store_true"); a=ap.parse_args()
 if not a.allow_label_read: raise RuntimeError("requires --allow-label-read")
 if digest(LABELS)!=LABEL_HASH: raise RuntimeError("label hash mismatch")
 frames={}; ids=None
 for k,p in FILES.items():
  f=pd.read_parquet(p)[["ordinal","image_id",SCORE,"configuration_hash"]]
  if len(f)!=11841 or f.ordinal.tolist()!=list(range(11841)) or not f.image_id.is_unique or not np.isfinite(f[SCORE]).all(): raise RuntimeError(f"invalid {k}")
  if ids is None: ids=f.image_id.tolist()
  if f.image_id.tolist()!=ids: raise RuntimeError(f"ID order mismatch {k}")
  frames[k]=f
 registry={"scope":"completed clean plus user-selected production cells","score":SCORE,"score_direction":1,"label_sha256":LABEL_HASH,"scorer_sha256":digest(Path(__file__)),"files":{k:{"path":str(p.relative_to(ROOT)),"sha256":digest(p),"configuration_hash":str(frames[k].configuration_hash.iloc[0])} for k,p in FILES.items()},"note":"Config hash differs only because the runner gained a condition selector; model/preprocessing/metric configuration is unchanged."}
 rp=PROJECT/"results/selected_evaluation_registry.json"; atomic(registry,rp)
 labels=pd.read_parquet(LABELS); joined={k:f.merge(labels,on="image_id",validate="one_to_one",sort=False) for k,f in frames.items()}
 y=joined["clean"].label.to_numpy(); scores={k:f[SCORE].to_numpy() for k,f in joined.items()}; t=threshold(y,scores["clean"]); base=metrics(y,scores["clean"],t)
 groups=[np.flatnonzero(y==c) for c in (0,1)]; rng=np.random.default_rng(20260830); idxs=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(2000)]
 rows=[]
 for k,s in scores.items():
  m=metrics(y,s,t); boots=[]; diffs=[]
  for ix in idxs:
   a3=(roc_auc_score(y[ix],s[ix]),average_precision_score(y[ix],s[ix]),balanced_accuracy_score(y[ix],s[ix]>=t)); c3=(roc_auc_score(y[ix],scores['clean'][ix]),average_precision_score(y[ix],scores['clean'][ix]),balanced_accuracy_score(y[ix],scores['clean'][ix]>=t))
   boots.append(a3); diffs.append(np.subtract(a3,c3))
  b=np.asarray(boots); d=np.asarray(diffs)
  for i,n in enumerate(("auroc","average_precision","balanced_accuracy")):
   m[n+"_ci95"]=[float(x) for x in np.percentile(b[:,i],[2.5,97.5])]; m[n+"_change_from_clean"]=float(m[n]-base[n]); m[n+"_change_ci95"]=[float(x) for x in np.percentile(d[:,i],[2.5,97.5])]
  rows.append({"condition":k,**m})
 out={"status":"COMPLETE FOR FOUR SELECTED CONDITIONS; NOT THE FULL 16-CELL MATRIX","n":11841,"real":3998,"fake":7843,"primary_score":SCORE,"threshold":t,"threshold_status":"post-hoc full-clean oracle","bootstrap":{"replicates":2000,"seed":20260830,"paired":True,"stratified":True},"registry_sha256":digest(rp),"results":rows}
 op=PROJECT/"results/metrics/selected_production_metrics.json"; atomic(out,op); pd.DataFrame([{x:y for x,y in r.items() if not isinstance(y,list)} for r in rows]).to_csv(op.with_suffix('.csv'),index=False)
 print(pd.DataFrame(rows)[["condition","auroc","average_precision","accuracy","balanced_accuracy","f1","auroc_change_from_clean"]].to_string(index=False)); print(f"threshold={t} output={op} sha256={digest(op)}")
if __name__=="__main__": main()
