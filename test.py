from transformers import AutoTokenizer

t_small = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
t_large = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")

text = "量子力学是20世纪最重要的物理学分支之一。"
print(t_small.encode(text) == t_large.encode(text))      # 应为 True
print(t_small.vocab_size == t_large.vocab_size)          # 应为 True