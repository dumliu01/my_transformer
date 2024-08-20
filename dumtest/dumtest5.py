from torchtext.data import Field,Example,Dataset
from torchtext import vocab
import os


# 1.数据
corpus = ["D'aww! He matches this background colour",
         "Yo bitch Ja Rule is more succesful then",
         "If you have a look back at the source"]
labels = [0,1,0]
# 2.定义不同的Field
TEXT = Field(sequential=True, lower=True, fix_length=10,tokenize=str.split,batch_first=True)
LABEL = Field(sequential=False, use_vocab=False)
fields = [("comment", TEXT),("label",LABEL)]
# 3.将数据转换为Example对象的列表
examples = []
for text,label in zip(corpus,labels):
    example = Example.fromlist([text,label],fields=fields)
    examples.append(example)
print(type(examples[0]))
print(examples[0].comment)
print(examples[0].label)
# 4.构建词表
new_corpus = [example.comment for example in examples]
TEXT.build_vocab(new_corpus)
print(TEXT.process(new_corpus))
