import json
import time
from pathlib import Path

import torch

from syntrix.models import build_model
from syntrix.utils.device import get_device


def benchmark(model_name="cnn",batch_size=32,warmup=10,iterations=50):
    device=get_device(); model=build_model(model_name).to(device).eval(); x=torch.randn(batch_size,1,28,28,device=device)
    with torch.inference_mode():
        for _ in range(warmup): model(x)
        if device.type=="cuda": torch.cuda.synchronize()
        times=[]
        for _ in range(iterations):
            t=time.perf_counter(); model(x)
            if device.type=="cuda": torch.cuda.synchronize()
            times.append((time.perf_counter()-t)*1000)
    times.sort(); p50=times[len(times)//2]; p95=times[max(0,int(len(times)*.95)-1)]
    result={"model":model_name,"device":str(device),"batch_size":batch_size,"p50_ms":p50,"p95_ms":p95,"throughput_samples_per_sec":batch_size/(p50/1000)}
    Path("experiments").mkdir(exist_ok=True); Path("experiments/benchmark.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result
