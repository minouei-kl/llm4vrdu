from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import sys

dname = sys.argv[1]
expname = dname.replace('.json', '')
# dname = 'DeepForm-unk_template-train_200-test_159-valid_100-SD_0.json'
root = '/netscratch/minouei/sources/llm/few-exp/data/'
train_data = load_dataset("json", data_files=root+"train"+dname, split="train")


model_id = "codellama/CodeLlama-7b-Instruct-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=0,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# target_modules="all-linear"
target_modules = ['q_proj', 'k_proj', 'v_proj',
                  'o_proj', 'gate_proj', 'down_proj', 'up_proj']
peft_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.05,
    r=16,
    target_modules=target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)

args = TrainingArguments(
    output_dir=expname,
    report_to="tensorboard",
    num_train_epochs=100,
    max_steps=1000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=1e-4,
    ddp_find_unused_parameters=False,
    bf16=True,
    save_safetensors=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)


max_seq_length = 8192  # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    # eval_dataset=val_data,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    dataset_text_field="text",
    args=args,
)
# train
trainer.train()  # there will not be a progress bar since tqdm is disabled

# save model
trainer.save_model()

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
)

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained(expname+"merged_model", safe_serialization=True)
tokenizer.save_pretrained(expname+"merged_model")
