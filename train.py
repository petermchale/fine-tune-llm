import torch 
from datasets import load_dataset 
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training 
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

def train(): 
  print('Setting up model...')
  original_model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/xgen-7b-8k-base", 
    load_in_4bit=True, 
    torch_dtype=torch.float16,
    device_map="auto"
  )
  tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen-7b-8k-base", trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  original_model.resize_token_embeddings(len(tokenizer))
  original_model = prepare_model_for_int8_training(original_model)

  print('Applying LoRA ...')
  lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.05, 
    bias="none",
    task_type="CAUSAL_LM",
  ) 
  peft_model = get_peft_model(original_model, lora_config)

  print("Parameter-efficient fine-tuning (training)...")
  train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
  training_args = TrainingArguments(
    output_dir="xgen-7b-tuned-alpaca",
    per_device_train_batch_size=4, 
    optim='adamw_torch',
    logging_steps=100, 
    learning_rate=2e-4, 
    fp16=True, 
    warmup_ratio=0.1,
    lr_scheduler_type="linear", 
    num_train_epochs=0.15, 
    save_strategy="epoch",
    push_to_hub=True,
  )
  trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=1024, 
    tokenizer=tokenizer,
    args=training_args,
    packing=True,
    peft_config=lora_config,
  )  
  trainer.train()

  print("Pushing LoRA adapators to Hub...")
  trainer.push_to_hub()

  print("Done!")

if __name__ == "__main__":
  train()