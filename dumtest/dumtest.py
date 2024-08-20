import torch
a = torch.cuda.is_available()
print(a)

from torchtext.data import Field, BucketIterator
import spacy
import data
from torchtext.datasets.translation import Multi30k

# 加载Spacy的英文模型，如果是其他语言，可以加载对应的模型
nlp = spacy.load('en_core_web_sm')


# 定义自定义的分词函数
def tokenize(text):
    doc = nlp(text)

    source = Field(tokenize=nlp, init_token='<sos>', eos_token='<eos>',
                        lower=True, batch_first=True)
    source.build_vocab(data.train_data, min_freq=2)

    train_data = Multi30k.splits(exts=('.en'), fields=(source))

    # 返回只包含单词的tokens列表
    return train_data


# 定义Field，并使用自定义的tokenizer
TEXT = Field(tokenize=tokenize, lower=True)  # lower=True表示转换为小写

# 假设我们有一些示例文本
examples = ["The quick brown fox jumps over the lazy dog.", "Another sentence to tokenize!"]

# 将文本转换为torchtext的Example对象
examples_data = [TEXT.preprocess(example) for example in examples]

# 打印转换后的tokens
for example in examples_data:
    print(example)

