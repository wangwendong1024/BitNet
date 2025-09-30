from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")  # 自动分配到 GPU

prompt = "请生成完整 Python 程序，实现快速排序。"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,  # 输出长度
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
