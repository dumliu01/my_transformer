import torch
import jieba
from torchtext import data


def tokenizer(text):
    return jieba.lcut(text)


TEXT = data.Field(sequential=True, tokenize=tokenizer)
LABEL = data.Field(sequential=False, use_vocab=False)

label_to_index = {
    'neural': 0,
    'happy': 1,
    'angry': 2,
    'sad': 3,
    'fear': 4,
    'surprise': 5,
}

import json


def get_dataset(json_file, text_field, label_field, test=False):
    if test:  # 如果为test，则没有label
        label_field = None
    # 定义fields
    fields = [('text', text_field), ('label', label_field)]

    # 读取文件
    with open(json_file, 'r') as rf:
        filedata = json.load(rf)

    # 生成examples
    examples = []
    for i in filedata:
        text = i.get('content', None)
        label = i.get('label', None)
        if not text:
            continue
        label_index = label_to_index.get(label, None)  # 得到数字label
        examples.append(data.Example.fromlist([text, label_index], fields))

    return data.Dataset(examples, fields)


train_data = get_dataset('./data/train/usual_train.txt', TEXT, LABEL)

TEXT.build_vocab(train_data)
# print(TEXT.vocab.__dict__)

train_iter = data.BucketIterator(dataset=train_data, batch_size=64, shuffle=True, sort_within_batch=False, repeat=False)

# 定义LSTM
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

weight_matrix = TEXT.vocab.vectors


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.word_embedding = nn.Embedding(len(TEXT.vocab), 300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1)
        self.decoder = nn.Linear(128, 6)

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        # print(embeds.shape)
        lstm_out = self.lstm(embeds)[0]
        # print(lstm_out.shape)
        final = lstm_out[-1]
        y = self.decoder(final)
        return y


model = LSTM()
model.train()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
crition = F.cross_entropy

print('Start training...')
EPOCHES = 20
for epoch in range(1, EPOCHES + 1):
    losses = 0
    for batch in train_iter:
        optimizer.zero_grad()
        predicted = model(batch.text)
        # print(predicted, batch.label)
        loss = crition(predicted, batch.label)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    print('Epoch: %d | Loss: %f' % (epoch, losses / (len(train_iter))))