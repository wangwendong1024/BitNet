from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 模型路径
model_path = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# 加载 tokenizer 和模型（自动分配 GPU）
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Prompt，要求完整生成程序
prompt = """
你是一个非常有帮助的助手，请完整生成一个Python程序，实现快速排序。
代码必须完整，不要省略任何部分。
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成参数
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,    # 输出长度，可调大
    do_sample=True,         # 随机采样
    temperature=0.7,        # 温度
    top_p=0.9,              # nucleus sampling
    repetition_penalty=1.1  # 防止重复
)

# 解码输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 输出到控制台
print(generated_text)

# 保存到文件
with open("output_program.py", "w", encoding="utf-8") as f:
    f.write(generated_text)
