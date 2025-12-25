import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os 
import sys 

LORA_PATH = "llama-3.1-wikipedia-improved/checkpoint-1800"
BASE_MODEL_LOCAL = "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
BASE_MODEL_HF = "meta-llama/Llama-3.1-8B-Instruct"

model, tokenizer = None, None
gpu = torch.cuda.is_available()

def get_base_model_path():
    """Check if base model exists locally, otherwise download from HuggingFace"""
    if os.path.exists(BASE_MODEL_LOCAL):
        return BASE_MODEL_LOCAL, True
    else:
        print(f"Local model not found. Will download from HuggingFace: {BASE_MODEL_HF}")
        print("This is a one-time download (~16GB). Please wait...")
        return BASE_MODEL_HF, False

def load_model():
    """Load Llama model with LoRA adapters"""
    global model, tokenizer
    
    use_gpu = torch.cuda.is_available()
    base_model_path, local_only = get_base_model_path()

    try:
        tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, local_files_only=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=local_only)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_gpu:
        # 4-bit quantization for GPU memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="cuda:0",
            torch_dtype=torch.float16,
            local_files_only=local_only,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            local_files_only=local_only,
            trust_remote_code=True,
        )

    try:
        model = PeftModel.from_pretrained(
            base_model, 
            LORA_PATH,
            torch_dtype=torch.float16,
            is_trainable=False,
        )
        model.eval()
    except Exception as e:
        model = base_model
        model.eval()

def generate_rag(context, history, query, max_tokens=250):
    """Generate response using context and conversation history"""
    
    hist_str = ""
    for msg in history[-4:]:
        role = "user" if msg["role"] == 'user' else "assistant"
        hist_str += f"{role}: {msg['content']}\n"

    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Use the context below to answer questions.

CONTEXT:
{context}

HISTORY:
{hist_str}

<|eot_id|><|start_header_id|>user<|end_header_id|>
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""

    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


if __name__ == '__main__':
    load_model()
    test_context = "Artificial Intelligence (AI) is intelligence demonstrated by machines."
    print(generate_rag(test_context, [], "What is AI?"))

