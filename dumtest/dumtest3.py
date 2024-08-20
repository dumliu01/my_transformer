import torch
from torchtext.data import Field, BucketIterator, LabelField

# 定义分词函数，这里使用简单的split方法
def tokenize(text):
    return text.split()

# 创建文本和标签的Field对象
TEXT = Field(tokenize=tokenize, lower=True, include_lengths=True)
LABEL = LabelField(dtype=torch.float)  # 假设标签是浮点数

# 假设有一些示例数据
examples = ["The quick brown fox jumps over the lazy dog.", "Another sentence to tokenize!"]

# 将示例数据转换为Field可以处理的格式
for example in examples:
    # 这里需要将文本和标签封装成一种特定的格式
    # 由于我们只有文本，所以这里只是演示如何使用Field
    TEXT.build_vocab([example])  # 构建词汇表

# 打印词汇表
print(TEXT.vocab.stoi)  # stoi是string to index的缩写，即字符串到索引的映射

# 将文本转换为tokens
tokens = [TEXT.process(example) for example in examples]

# 打印tokens
print(tokens)

# 将tokens转换为数值索引
text_data = [torch.tensor([TEXT.vocab.stoi[w] for w in tokens_line], dtype=torch.int64) for tokens_line in tokens]

# 打印数值索引
print(text_data)