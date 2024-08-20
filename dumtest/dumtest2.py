from torchtext.data import Field, BucketIterator
import spacy
import data
from torchtext.datasets.translation import Multi30k
from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

def string_to_encoded_iterator(text, tokenizer, vocab, init_token, eos_token):
    """
    将输入的字符串通过分词器分词，并使用词汇表编码，然后创建一个迭代器。

    参数:
    - text: 输入的字符串。
    - tokenizer: 分词器函数，用于将字符串分割成单词或标记。
    - vocab: 词汇表对象，用于将单词或标记转换为整数索引。
    - init_token: 序列的起始标记。
    - eos_token: 序列的结束标记。

    返回:
    - encoded_iterator: 一个生成整数索引的迭代器。
    """

    # 使用分词器对文本进行分词
    tokens = tokenizer(text)

    # 将分词结果转换为词汇表中的索引，并添加起始和结束标记
    encoded_sequence = [vocab(init_token)] + [vocab(token) for token in tokens] + [vocab(eos_token)]

    # 创建一个迭代器，用于生成编码后的序列
    encoded_iterator = iter(encoded_sequence)

    return encoded_iterator

# 示例用法：
# 假设我们已经有了一个分词器和一个词汇表，以及起始和结束标记
# tokenizer = get_tokenizer('spacy')  # 示例分词器
# vocab = build_vocab_from_iterator(...)  # 示例词汇表构建函数
# init_token = '<sos>'
# eos_token = '<eos>'


tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

train, valid, test = loader.make_dataset()
loader.build_vocab(train_data=train, min_freq=2)

tokenizer = spacy.load('en_core_web_sm')
source = Field(tokenize=tokenizer, init_token='<sos>', eos_token='<eos>',
                        lower=True, batch_first=True)
source.build_vocab(train, min_freq=2)
#vocab = source.vocab
#vocab.set_default_index(source_vocab[self.eos_token])

#init_token = '<sos>'
#eos_token = '<eos>'

# 输入的文本字符串
text = "This"
t = source.process(text)

print(t)
# 调用函数，获取编码的迭代器
#encoded_iter = string_to_encoded_iterator(text, tokenizer, vocab, init_token, eos_token)

# 使用迭代器
#for index in encoded_iter:
#    print(index, end=' ')  # 打印每个索引值