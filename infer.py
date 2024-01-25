from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
import torch
from peft import AutoPeftModelForCausalLM
import json
import os

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
system_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. your response is a JSON object that appropriately completes the request. The JSON must be between [JSON] and [/JSON] tags. """


dname = 'cord.json'
expname = dname.replace('.json', '')  # +'merged_model'
# dname = 'DeepForm-unk_template-train_200-test_159-valid_100-SD_0.json'
root = '/netscratch/minouei/sources/llm/few-exp/cord/'
dataset = load_dataset("json", data_files=root+"test"+dname, split="train")


model = AutoPeftModelForCausalLM.from_pretrained(
    expname,
    # torch_dtype=torch.float16,
    device_map=0, trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(expname)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def generate_response(model, tokenizer, sample, temperature=0.9):

    inst = f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n"
    prompt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{inst} {E_INST}"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=1500,  # top_k=10,
                                 length_penalty=1.05, repetition_penalty=1.1, num_beams=2,
                                 num_return_sequences=model.config.num_return_sequences,
                                 early_stopping=model.config.early_stopping,
                                 eos_token_id=model.config.eos_token_id,
                                 pad_token_id=model.config.pad_token_id,
                                 do_sample=True, top_p=0.99, temperature=temperature)

    response = tokenizer.batch_decode(outputs.detach().cpu(
    ).numpy(), skip_special_tokens=True)[0][len(prompt):]

    return response


def jsonl_to_list(jsonl_file):
    json_objects = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            json_objects.append(json.loads(line))
    return json_objects


json_objects = []
json_file_path = "res/"+dname
if os.path.exists(json_file_path):
    try:
        json_objects = jsonl_to_list(json_file_path)
    except:
        pass

allres = []
# for sample in dataset.select(range(22)):
for sample in dataset:
    fn = False
    for item in json_objects:
        if item.get('filename') == sample['filename'] and item.get('page_id') == sample['page_id']:
            fn = True
            break
    if fn:
        continue

    try:
        response1 = generate_response(model, tokenizer, sample)
    except:
        response1 = '{}'

    res = {'filename': sample['filename'],  # 'page_id': sample['page_id'],
           'response': response1, 'gt': sample['output']}
    # allres.append({'filename': sample['filename'], 'page_id': sample['page_id'],
    #               'response': response1, 'gt': sample['output']})
    print(f"response:\n{response1}\n")
    print(f"Ground truth:\n{sample['output']}\n")

    with open(json_file_path, "a") as json_file:
        json.dump(res, json_file)
        json_file.write('\n')

# model = AutoPeftModelForCausalLM.from_pretrained(
#     output_dir,
#     low_cpu_mem_usage=True,
# )

# # Merge LoRA and base model
# merged_model = model.merge_and_unload()

# # Save the merged model
# merged_model.save_pretrained("merged_model", safe_serialization=True)
# tokenizer.save_pretrained("merged_model")
