import torch

from models.model.transformer import Transformer
from data import *
from util.bleu import idx_to_word, get_bleu
from torchtext.data import Field, BucketIterator


def predict(input):
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

    model.load_state_dict(torch.load('saved/model-1.pt'))
    model.eval()

    src = input
    trg = ('1')
    output = model(src, trg[:, :-1])
    return output



def translate(sentence):
    # 对源句子进行编码
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

    model.load_state_dict(torch.load('saved/model-1.pt'))
    model.eval()


    output_result = ""
    with torch.no_grad():
        batch_bleu = []
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
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
                    output_result = output_words
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

    return output_result


if __name__ == '__main__':
    output = translate("hello")
    print(output)

