import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from syntrix.data.vision import make_fashion_mnist
from syntrix.models import build_model
from syntrix.utils.device import get_device
from syntrix.utils.repro import seed_everything


def train(model_name="cnn", epochs=2, batch_size=128, lr=3e-4, seed=42, amp=True):
    seed_everything(seed); device=get_device()
    train_loader,test_loader=make_fashion_mnist(Path("data"),batch_size)
    model=build_model(model_name).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss()
    scaler=torch.amp.GradScaler("cuda",enabled=(amp and device.type=="cuda")); history=[]; start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); total=n=0
        for x,y in train_loader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16 if device.type=="cuda" else torch.bfloat16,enabled=amp):
                loss=loss_fn(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()*x.size(0); n+=x.size(0)
        m=evaluate_model(model,test_loader,device); rec={"epoch":epoch,"train_loss":total/n,**m}; history.append(rec); print(rec)
    Path("artifacts").mkdir(exist_ok=True); torch.save({"model":model.state_dict(),"model_name":model_name},"artifacts/model.pt")
    Path("experiments").mkdir(exist_ok=True); result={"model":model_name,"device":str(device),"epochs":epochs,"elapsed_seconds":time.perf_counter()-start,"history":history}
    Path("experiments/metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result

def evaluate_model(model,loader,device):
    model.eval(); ys=[]; preds=[]
    with torch.inference_mode():
        for x,y in loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist()); ys.extend(y.tolist())
    return {"accuracy":accuracy_score(ys,preds),"f1_macro":f1_score(ys,preds,average="macro")}

def evaluate_saved():
    device=get_device(); _,loader=make_fashion_mnist(Path("data"),256); ckpt=torch.load("artifacts/model.pt",map_location=device,weights_only=True); model=build_model(ckpt["model_name"]).to(device); model.load_state_dict(ckpt["model"]); m=evaluate_model(model,loader,device); print(json.dumps(m,indent=2)); return m
