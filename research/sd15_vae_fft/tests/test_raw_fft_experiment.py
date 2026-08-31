import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"sb15_fft"))
from sb15_fft.spectral_metrics import image_metrics

def direct(x):
    gray=x.mean(1); fft=torch.fft.rfft2(gray,norm="ortho").abs().square()
    fy=torch.fft.fftfreq(512).abs()[:,None]; fx=torch.fft.rfftfreq(512).abs()[None,:]
    radius=torch.sqrt(fx.square()+fy.square())/(.5**2+.5**2)**.5
    means=[fft[:,m].mean(1) for m in (radius<.25,(radius>=.25)&(radius<.5),radius>=.5)]
    return (means[2]/sum(means).clamp_min(1e-12)).numpy()

def cases():
    z=torch.zeros(1,3,512,512); w=torch.ones_like(z); impulse=z.clone(); impulse[:,:,256,256]=1
    grad=torch.linspace(0,1,512).view(1,1,1,512).expand(1,3,512,512)
    rand=torch.rand((3,3,512,512),generator=torch.Generator().manual_seed(42))
    return [z,w,impulse,grad,rand]

def test_parity_finite_batch_order_and_distinct_ratios():
    for x in cases():
        got=image_metrics(x,torch.zeros_like(x)); np.testing.assert_allclose(got["fft_high_bandmean_ratio"],direct(x),rtol=1e-7,atol=1e-7); assert all(np.isfinite(v).all() for v in got.values())
    x=cases()[-1]; batch=image_metrics(x,torch.zeros_like(x))
    singles=np.concatenate([image_metrics(v[None],torch.zeros_like(v[None]))["fft_high_bandmean_ratio"] for v in x])
    np.testing.assert_allclose(batch["fft_high_bandmean_ratio"],singles,rtol=1e-7,atol=1e-7)
    assert not np.allclose(batch["fft_high_bandmean_ratio"],batch["fft_high_coeffsum_ratio"])

def test_frozen_inventories():
    m=pd.read_parquet(ROOT/"sb15_fft/results/manifests/frozen_extraction.parquet"); l=pd.read_parquet(ROOT/"data/manifests/wildfake_test_labels.parquet")
    assert len(m)==11841 and m.image_id.is_unique and m.ordinal.tolist()==list(range(11841))
    assert len(l)==11841 and l.image_id.is_unique and l.label.value_counts().to_dict()=={1:7843,0:3998}

