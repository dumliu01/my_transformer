import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from torchtext.data import Field, BucketIterator

tokenizer = Tokenizer()

def string2token(text):
    init_token = '<sos>'
    eos_token = '<eos>'
    source = Field(tokenize=tokenizer.tokenize_en, init_token=init_token, eos_token=eos_token,
                        lower=True, batch_first=True)
    target = Field(tokenize=tokenizer.tokenize_de, init_token=init_token, eos_token=eos_token,
                        lower=True, batch_first=True)
    l = [text]
    source.build_vocab(train, min_freq=2)
    t = source.process(l).cuda()
    target.build_vocab(train, min_freq=2)
    t2 = target.process(l).cuda()

    return t,t2

def test_model(num_examples):
    #iterator = test_iter

    model = Transformer(src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        trg_sos_idx=trg_sos_idx,
                        d_model=d_model,
                        enc_voc_size=enc_voc_size,
                        dec_voc_size=dec_voc_size,
                        max_len=max_len,
                        ffn_hidden=ffn_hidden,
                        n_head=n_heads,
                        n_layers=n_layers,
                        drop_prob=drop_prob,
                        device=device).to(device)
    model.load_state_dict(torch.load("./saved/model-2.pt"))

    with torch.no_grad():
        batch_bleu = []
        #for i, batch
        t,t2 = string2token("hello")
        src = t
        trg = t2
        output = model(src, trg[:, :-1])

        total_bleu = []
        for j in range(num_examples):
            try:
                src_words = idx_to_word(src[j], loader.source.vocab)
                trg_words = idx_to_word(trg[j], loader.target.vocab)
                output_words = output[j].max(dim=1)[1]
                output_words = idx_to_word(output_words, loader.target.vocab)

                print('source :', src_words)
                print('target :', trg_words)
                print('predicted :', output_words)
                print()
                bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                total_bleu.append(bleu)
            except:
                pass

        total_bleu = sum(total_bleu) / len(total_bleu)
        print('BLEU SCORE = {}'.format(total_bleu))
        batch_bleu.append(total_bleu)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        print('TOTAL BLEU SCORE = {}'.format(batch_bleu))

if __name__ == '__main__':
    test_model(1)
