"""Optional LoRA/PEFT smoke test. Install: pip install -e '.[research]'"""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2)
config=LoraConfig(task_type=TaskType.SEQ_CLS,r=8,lora_alpha=16,lora_dropout=.05,target_modules=["q_lin","v_lin"])
model=get_peft_model(model,config); model.print_trainable_parameters()
