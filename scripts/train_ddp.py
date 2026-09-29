import argparse,os,torch,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from torchvision import datasets,transforms
from syntrix.models import build_model
from syntrix.utils.repro import seed_everything

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=2); args=ap.parse_args(); distributed=int(os.environ.get("WORLD_SIZE","1"))>1
    if distributed: dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo"); rank=dist.get_rank(); local=int(os.environ.get("LOCAL_RANK","0"))
    else: rank,local=0,0
    seed_everything(42+rank); device=torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    ds=datasets.FashionMNIST("data",train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.2860,),(.3530,))])); sampler=DistributedSampler(ds) if distributed else None; loader=DataLoader(ds,batch_size=128,sampler=sampler,shuffle=sampler is None)
    model=build_model("cnn").to(device); model=DDP(model,device_ids=[local] if device.type=="cuda" else None) if distributed else model; opt=torch.optim.AdamW(model.parameters(),lr=3e-4); loss_fn=torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        model.train()
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); loss=loss_fn(model(x),y); loss.backward(); opt.step()
        if rank==0: print({"epoch":epoch+1,"loss":float(loss)})
    if distributed: dist.destroy_process_group()
if __name__=="__main__": main()
