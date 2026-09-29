import argparse,json
from syntrix.utils.device import describe_device
from syntrix.training.engine import train,evaluate_saved
from syntrix.evaluation.benchmark import benchmark

def main():
    p=argparse.ArgumentParser(prog="syntrix"); sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("device"); t=sub.add_parser("train"); t.add_argument("--model",choices=["cnn","vit"],default="cnn"); t.add_argument("--epochs",type=int,default=2); t.add_argument("--batch-size",type=int,default=128); t.add_argument("--lr",type=float,default=3e-4); t.add_argument("--seed",type=int,default=42); t.add_argument("--no-amp",action="store_true")
    sub.add_parser("evaluate"); b=sub.add_parser("benchmark"); b.add_argument("--model",choices=["cnn","vit"],default="cnn"); b.add_argument("--batch-size",type=int,default=32); sub.add_parser("serve")
    a=p.parse_args()
    if a.cmd=="device": print(json.dumps(describe_device(),indent=2))
    elif a.cmd=="train": train(a.model,a.epochs,a.batch_size,a.lr,a.seed,not a.no_amp)
    elif a.cmd=="evaluate": evaluate_saved()
    elif a.cmd=="benchmark": benchmark(a.model,a.batch_size)
    else:
        import uvicorn; uvicorn.run("syntrix.inference.api:app",host="127.0.0.1",port=8000)
if __name__=="__main__": main()
