import torch
from torchtext.data import Field, LabelField, BucketIterator

def tokenize(text):
    return text.split()

# 创建文本字段
TEXT = Field(tokenize=tokenize, lower=True, include_lengths=True)

# 创建标签字段，这里假设标签是分类任务
LABEL = LabelField(dtype=torch.float)  # 根据你的任务选择合适的dtype

# 假设有以下文本数据
train_data = ["Hello World", "Hello TorchText", "TorchText is great"]

# 构建词汇表
TEXT.build_vocab(train_data)

# 处理文本数据
processed_examples = [TEXT.process(text) for text in train_data]

from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [TEXT.process(text) for text in texts]
        self.labels = labels#[LABEL.process(label) for label in labels]

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.texts)


# 假设有对应的标签数据
train_labels = [1, 0, 1]

# 创建数据集实例
dataset = TextDataset(train_data, train_labels)

# 创建批次迭代器
train_iter = BucketIterator(dataset=dataset, batch_size=2, sort_within_batch=True, sort_key=lambda x: len(x[0]))

for batch in train_iter:
    texts, labels = batch.text, batch.label

    # 这里添加模型训练代码
    # ...
    print(texts)
    print(labels)