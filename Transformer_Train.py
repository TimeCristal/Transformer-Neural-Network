# %%
from transformer import Transformer  # this is the transformer.py file
import torch
# %%
import numpy as np

#%%
Present_file = 'dataset/Present.txt'
Future_file = 'dataset/Future.txt'

# Generated this by filtering Appendix code

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

Future_vocabulary = [START_TOKEN, ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', PADDING_TOKEN, END_TOKEN]

Present_vocabulary = Future_vocabulary

#%%
index_to_Future = {k: v for k, v in enumerate(Future_vocabulary)}
Future_to_index = {v: k for k, v in enumerate(Future_vocabulary)}
index_to_Present = {k: v for k, v in enumerate(Present_vocabulary)}
Present_to_index = {v: k for k, v in enumerate(Present_vocabulary)}
#%%
with open(Present_file, 'r') as file:
    Present_sentences = file.readlines()
with open(Future_file, 'r') as file:
    Future_sentences = file.readlines()

# Limit Number of sentences
TOTAL_SENTENCES = 200000
Present_sentences = Present_sentences[:TOTAL_SENTENCES]
Future_sentences = Future_sentences[:TOTAL_SENTENCES]
Present_sentences = [sentence.rstrip('\n').lower() for sentence in Present_sentences]
Future_sentences = [sentence.rstrip('\n') for sentence in Future_sentences]
#%%
Present_sentences[:10]
#%%
Future_sentences[:10]
#%%
import numpy as np

PERCENTILE = 97
print(f"{PERCENTILE}th percentile length Future: {np.percentile([len(x) for x in Future_sentences], PERCENTILE)}")
print(f"{PERCENTILE}th percentile length Present: {np.percentile([len(x) for x in Present_sentences], PERCENTILE)}")

#%%
max_sequence_length = 40


def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True


def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1)  # need to re-add the end token so leaving 1 space


valid_sentence_indicies = []
for index in range(len(Future_sentences)):
    Future_sentence, Present_sentence = Future_sentences[index], Present_sentences[index]
    if is_valid_length(Future_sentence, max_sequence_length) \
            and is_valid_length(Present_sentence, max_sequence_length) \
            and is_valid_tokens(Future_sentence, Future_vocabulary):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(Future_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")
#%%
Future_sentences = [Future_sentences[i] for i in valid_sentence_indicies]
Present_sentences = [Present_sentences[i] for i in valid_sentence_indicies]
#%%
Future_sentences[:3]
#%%
import torch

num_epochs = 1000

d_model = 512 * 2
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 4  #1
max_sequence_length = 40  #200
kn_vocab_size = len(Future_vocabulary)

transformer = Transformer(d_model,
                          ffn_hidden,
                          num_heads,
                          drop_prob,
                          num_layers,
                          max_sequence_length,
                          kn_vocab_size,
                          Present_to_index,
                          Future_to_index,
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)
#%%
transformer
#%%
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):

    def __init__(self, Present_sentences, Future_sentences):
        self.Present_sentences = Present_sentences
        self.Future_sentences = Future_sentences

    def __len__(self):
        return len(self.Present_sentences)

    def __getitem__(self, idx):
        return self.Present_sentences[idx], self.Future_sentences[idx]


#%%
dataset = TextDataset(Present_sentences, Future_sentences)
#%%
len(dataset)
#%%
dataset[1]
#%%
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)
#%%
for batch_num, batch in enumerate(iterator):
    print(batch)
    if batch_num > 3:
        break
#%%
from torch import nn

criterian = nn.CrossEntropyLoss(ignore_index=Future_to_index[PADDING_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#%%
NEG_INFTY = -1e9


def create_masks(eng_batch, kn_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
        eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
        kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


#%% md
# Modify mask such that the padding tokens cannot look ahead.
# In Encoder, tokens before it should be -1e9 while tokens after it should be -inf.
#  
#%% md
# Note the target mask starts with 2 rows of non masked items: https://github.com/SamLynnEvans/Transformer/blob/master/Beam.py#L55
# 
#%%
transformer.train()
transformer.to(device)
total_loss = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, kn_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch,
                                                                                                              kn_batch)
        optim.zero_grad()
        kn_predictions = transformer(eng_batch,
                                     kn_batch,
                                     encoder_self_attention_mask.to(device),
                                     decoder_self_attention_mask.to(device),
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
        loss = criterian(
            kn_predictions.view(-1, kn_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == Future_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        sample_idx = np.random.randint(0, batch_size - 1)
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"Present: {eng_batch[sample_idx]}")
            print(f"Future Translation: {kn_batch[sample_idx]}")
            kn_sentence_predicted = torch.argmax(kn_predictions[sample_idx], axis=1)
            predicted_sentence = ""
            for idx in kn_sentence_predicted:
                if idx == Future_to_index[END_TOKEN]:
                    break
                predicted_sentence += index_to_Future[idx.item()]
            print(f"Future Prediction : {predicted_sentence}")

            transformer.eval()
            kn_sentence = ("",)
            # eng_sentence = ("should we go to the mall?",)#405 763543
            eng_sentence = ("405 763543",)
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                    eng_sentence, kn_sentence)
                predictions = transformer(eng_sentence,
                                          kn_sentence,
                                          encoder_self_attention_mask.to(device),
                                          decoder_self_attention_mask.to(device),
                                          decoder_cross_attention_mask.to(device),
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=True,
                                          dec_end_token=False)
                next_token_prob_distribution = predictions[0][word_counter]  # not actual probs
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_Future[next_token_index]
                kn_sentence = (kn_sentence[0] + next_token,)
                if next_token == END_TOKEN:
                    break

            print(f"Evaluation translation (405 763543) : {kn_sentence}")
            print("-------------------------------------------")
#%% md
# ## Inference
#%%
transformer.eval()

import torch


def translate(eng_sentence, transformer, max_sequence_length, index_to_Future, Future_to_index, temperature=1.0,
              top_k=5):
    eng_sentence = (eng_sentence,)
    kn_sentence = ("",)
    for word_counter in range(max_sequence_length):
        # Create masks for the transformer model
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            eng_sentence, kn_sentence)

        # Get predictions from the transformer model
        predictions = transformer(eng_sentence,
                                  kn_sentence,
                                  encoder_self_attention_mask.to(device),
                                  decoder_self_attention_mask.to(device),
                                  decoder_cross_attention_mask.to(device),
                                  enc_start_token=False,
                                  enc_end_token=False,
                                  dec_start_token=True,
                                  dec_end_token=False)

        # Apply temperature scaling
        next_token_logits = predictions[0][word_counter] / temperature

        # Apply top-k sampling
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
        top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
        next_token_index = torch.multinomial(top_k_probs, 1).item()

        next_token = index_to_Future[top_k_indices[next_token_index].item()]

        # Append the next token to the output sentence
        kn_sentence = (kn_sentence[0] + next_token,)

        # Check for the end token
        if next_token == END_TOKEN:
            break

    return kn_sentence[0]

def translate_old(eng_sentence):
    eng_sentence = (eng_sentence,)
    kn_sentence = ("",)
    for word_counter in range(max_sequence_length):
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            eng_sentence, kn_sentence)
        predictions = transformer(eng_sentence,
                                  kn_sentence,
                                  encoder_self_attention_mask.to(device),
                                  decoder_self_attention_mask.to(device),
                                  decoder_cross_attention_mask.to(device),
                                  enc_start_token=False,
                                  enc_end_token=False,
                                  dec_start_token=True,
                                  dec_end_token=False)
        next_token_prob_distribution = predictions[0][word_counter]
        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = index_to_Future[next_token_index]
        kn_sentence = (kn_sentence[0] + next_token,)
        if next_token == END_TOKEN:
            break
    return kn_sentence[0]

#%%
# translation = translate("405 763543")
# print(translation)
# print("real 33 54")

