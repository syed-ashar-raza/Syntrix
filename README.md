Syntrix-v0.1.0\\README.md

\# Syntrix



\*\*Syntrix — Multimodal AI Research \& Engineering Platform\*\*



A proof-oriented AI/ML engineering platform focused on reproducible computer-vision training, streaming data, model evaluation, inference benchmarking, production inference, observability, and parameter-efficient fine-tuning.



> \*\*Evidence note:\*\* Results below are measurements from the current local CPU environment. CUDA/GPU and distributed-training paths are implemented as engineering extensions but are not claimed as locally executed evidence.



\---



\## Architecture



```text

Dataset / Streaming Source

&#x20;         |

&#x20;         v

DataLoader / IterableDataset

&#x20;         |

&#x20;         v

CNN / TinyViT

&#x20;         |

&#x20;         v

Training + Reproducibility

&#x20;         |

&#x20;         +--> Evaluation

&#x20;         +--> Checkpointing

&#x20;         +--> Experiment Metrics

&#x20;         |

&#x20;         v

Inference Benchmarking

&#x20;         |

&#x20;         v

FastAPI Inference

&#x20;         |

&#x20;         +--> Validation

&#x20;         +--> Prediction

&#x20;         +--> Health

&#x20;         +--> Prometheus Metrics

&#x20;         |

&#x20;         v

Production-style Serving



Research Extension

&#x20;         |

&#x20;         v

Transformers --> PEFT / LoRA

Engineering Focus



Syntrix is designed to demonstrate practical AI engineering across:



Computer vision

Deep learning with PyTorch

Reproducible experimentation

Efficient data loading

Streaming datasets

Model evaluation

Inference performance

Production-style model serving

API validation

Observability

Parameter-efficient fine-tuning

Research-oriented benchmarking

Implemented

PyTorch CNN baseline

Compact TinyViT vision model

Fashion-MNIST training/evaluation pipeline

Deterministic reproducibility

Checkpoint save/load

Streaming IterableDataset

Device-aware data loading

CUDA-aware AMP path

Accuracy and macro-F1 evaluation

p50/p95 inference benchmarking

Throughput measurement

FastAPI inference API

Input validation

Prometheus request and latency metrics

PEFT/LoRA smoke test

Automated tests

Ruff linting

Docker support

PyTorch DDP training entry point

Verified Experimental Results



All results below were measured locally on a CPU-only environment.



CNN Baseline

Metric	Result

Model	CNN

Epochs	1

Batch size	256

Device	CPU

Train loss	1.057022

Accuracy	77.36%

Macro F1	76.54%

Training time	130.34 s

Seed	42

TinyViT Baseline

Metric	Result

Model	TinyViT

Epochs	1

Batch size	256

Device	CPU

Train loss	0.926349

Accuracy	77.41%

Macro F1	77.45%

Training time	306.16 s

Model Comparison



TinyViT achieved a slightly higher macro-F1 than the CNN in the measured one-epoch experiment:



CNN macro-F1: 76.54%

TinyViT macro-F1: 77.45%



The TinyViT training run took substantially longer on the CPU environment.



This comparison demonstrates an explicit model-selection trade-off between training cost and measured validation performance.



Inference Benchmark



Benchmark configuration:



Batch size: 32

Warm-up iterations: 10

Measurement iterations: 50

Device: CPU

Model	p50 Latency	p95 Latency	Throughput

CNN	16.04 ms	26.34 ms	1,995 samples/s

TinyViT	16.42 ms	26.14 ms	1,948 samples/s



The benchmark measures inference latency and throughput using the same CPU environment.



The results demonstrate how model architecture can affect latency and throughput even when measured inference latency is similar.



Reproducibility



CNN training was repeated with:



model: cnn

epochs: 1

batch size: 256

seed: 42



Both runs produced exactly:



train\_loss = 1.0570220036506652

accuracy   = 0.7736

f1\_macro   = 0.7653591722458049



This verifies deterministic behavior for the tested training configuration.



Runtime varied between runs, which is expected on a shared CPU execution environment.



Streaming Data



The streaming demonstration produced:



streamed\_samples: 1024

dataset\_materialized: False



This verifies lazy iteration without materializing the complete dataset in memory.



The demonstration uses a synthetic streaming source and therefore is an architectural proof rather than a large-scale production dataset benchmark.



Run it with:



python .\\scripts\\streaming\_demo.py

LoRA / PEFT



Syntrix includes an optional research stack using:



Hugging Face Transformers

PEFT

LoRA

Safetensors



A DistilBERT LoRA smoke test was successfully executed.



Verified Configuration

Metric	Result

Base model	DistilBERT

Total parameters	67,694,596

Trainable parameters	739,586

Trainable percentage	1.0925%

LoRA rank	8

LoRA alpha	16

LoRA dropout	0.05

Target modules	q\_lin, v\_lin



The result demonstrates parameter-efficient adapter construction with approximately 1.09% of model parameters trainable.



This is a LoRA configuration smoke test. It does not claim downstream fine-tuning accuracy because a complete fine-tuning experiment has not yet been executed.



Run it with:



python .\\scripts\\lora\_demo.py

Production-Style Inference API



Syntrix provides a FastAPI inference service.



Start the server:



python -m syntrix.cli serve



Server:



http://127.0.0.1:8000

Health Endpoint

GET /health



Verified response state:



status: ok

model\_loaded: True

device: cpu

Prediction Endpoint

POST /v1/predict



The endpoint validates that the request contains exactly 784 grayscale pixel values, corresponding to a 28 × 28 input.



Example:



$body = @{ values = @(0.0) \* 784 } | ConvertTo-Json -Compress



Invoke-RestMethod `

&#x20; -Method Post `

&#x20; -Uri http://127.0.0.1:8000/v1/predict `

&#x20; -ContentType "application/json" `

&#x20; -Body $body



A real inference request successfully returned:



class\_id   = 3

confidence = 0.44953712821006775



The example uses an all-zero synthetic input. It demonstrates the working inference path and API contract, not meaningful classification quality.



Prometheus Observability



The inference service exposes:



GET /metrics



Verified metrics include:



syntrix\_inference\_requests\_total

syntrix\_inference\_latency\_seconds



A real inference request produced:



requests: 1

latency: approximately 7.33 ms



The metrics provide the foundation for monitoring request volume and inference latency.



CLI

Device Information

python -m syntrix.cli device

Train CNN

python -m syntrix.cli train --model cnn --epochs 1 --batch-size 256 --seed 42

Train TinyViT

python -m syntrix.cli train --model vit --epochs 1 --batch-size 256

Evaluate Saved Checkpoint

python -m syntrix.cli evaluate

Benchmark CNN

python -m syntrix.cli benchmark --model cnn

Benchmark TinyViT

python -m syntrix.cli benchmark --model vit

Start API

python -m syntrix.cli serve

Testing



The current automated test suite passes:



4 passed in 3.58s



Run:



pytest -q



The test suite covers core model, streaming, and API behavior.



Code Quality



Syntrix uses Ruff for static analysis.



Run:



ruff check .

Installation



Create and activate the virtual environment:



python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1



Install development dependencies:



pip install -e ".\[dev]"



Install research dependencies:



pip install -e ".\[research]"



The research extra provides:



Transformers

PEFT

Safetensors

Hugging Face Hub dependencies

Project Structure

Syntrix/

│

├── configs/

│   └── base.yaml

│

├── scripts/

│   ├── lora\_demo.py

│   ├── streaming\_demo.py

│   └── train\_ddp.py

│

├── src/

│   └── syntrix/

│       ├── data/

│       │   ├── vision.py

│       │   └── streaming.py

│       │

│       ├── evaluation/

│       │   └── benchmark.py

│       │

│       ├── inference/

│       │   └── api.py

│       │

│       ├── models/

│       │   ├── cnn.py

│       │   └── vit.py

│       │

│       ├── training/

│       │   └── engine.py

│       │

│       ├── utils/

│       │   ├── device.py

│       │   └── repro.py

│       │

│       └── cli.py

│

├── tests/

│   ├── test\_api.py

│   ├── test\_models.py

│   └── test\_streaming.py

│

├── Dockerfile

├── pyproject.toml

├── README.md

└── .gitignore

Experiment Artifacts



Generated training artifacts are intentionally excluded from Git.



Training metrics:



experiments/metrics.json



Benchmark results:



experiments/benchmark.json



Model checkpoint:



artifacts/model.pt



Datasets:



data/



These files can be regenerated locally and are excluded to keep the repository lightweight.



Reproducible Experiment



Example:



python -m syntrix.cli train `

&#x20; --model cnn `

&#x20; --epochs 1 `

&#x20; --batch-size 256 `

&#x20; --seed 42



Then evaluate:



python -m syntrix.cli evaluate



Then benchmark:



python -m syntrix.cli benchmark --model cnn

Engineering Boundaries



The current verified environment is CPU-only.



Therefore this repository does not claim that the following have been executed locally:



CUDA GPU training

GPU mixed-precision benchmarking

multi-GPU training

multi-node distributed training

large-scale multimodal training

downstream LoRA fine-tuning evaluation



The project contains engineering paths for these research extensions, but they should only be considered demonstrated capabilities after they are actually executed and measured.



This distinction is intentional: Syntrix documents measured engineering evidence separately from implemented research paths.



Roadmap

Completed

&#x20;PyTorch vision training

&#x20;CNN baseline

&#x20;TinyViT baseline

&#x20;Deterministic reproducibility

&#x20;Model checkpointing

&#x20;Model evaluation

&#x20;Streaming dataset

&#x20;Inference benchmarking

&#x20;FastAPI inference

&#x20;Input validation

&#x20;Prometheus observability

&#x20;PEFT / LoRA smoke test

&#x20;Automated tests

&#x20;Ruff linting

&#x20;Docker support

&#x20;DDP training entry point

Research Extensions

&#x20;GPU benchmark

&#x20;GPU mixed-precision benchmark

&#x20;Multi-GPU distributed benchmark

&#x20;Full image-text multimodal training

&#x20;LoRA downstream fine-tuning

&#x20;Baseline/ablation experiment matrix

&#x20;Larger-scale dataset benchmark

License



Apache-2.0





After pasting and saving, \*\*don't commit yet\*\*. Run:



```powershell

git status --short

