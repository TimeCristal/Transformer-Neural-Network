# %%
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformer import Transformer  # this is the transformer.py file
from utils import load_checkpoint, TextDataset, create_masks, save_checkpoint, is_valid_length, is_valid_tokens

# Define a directory to save the checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

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
future_to_index = {v: k for k, v in enumerate(Future_vocabulary)}
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

PERCENTILE = 97
print(f"{PERCENTILE}th percentile length Future: {np.percentile([len(x) for x in Future_sentences], PERCENTILE)}")
print(f"{PERCENTILE}th percentile length Present: {np.percentile([len(x) for x in Present_sentences], PERCENTILE)}")

#%%
max_sequence_length = 40

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
# %%
save_every_n_epochs = 100
num_epochs = 1000

d_model = 512 * 2
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 4
max_sequence_length = 40
kn_vocab_size = len(Future_vocabulary)

transformer = Transformer(d_model,
                          ffn_hidden,
                          num_heads,
                          drop_prob,
                          num_layers,
                          max_sequence_length,
                          kn_vocab_size,
                          Present_to_index,
                          future_to_index,
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)
#%%
transformer

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


criterion = nn.CrossEntropyLoss(ignore_index=future_to_index[PADDING_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# %%
transformer.train()
transformer.to(device)
total_loss = 0

# try to load check point
start_epoch, best_loss = load_checkpoint(transformer, optim, checkpoint_dir)

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    epoch_loss = 0
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, kn_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            eng_batch,
            kn_batch,
            max_sequence_length)
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
        loss = criterion(
            kn_predictions.view(-1, kn_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == future_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
        # train_losses.append(loss.item())
        sample_idx = np.random.randint(0, batch_size - 1)
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"Present: {eng_batch[sample_idx]}")
            print(f"Future Translation: {kn_batch[sample_idx]}")
            kn_sentence_predicted = torch.argmax(kn_predictions[sample_idx], axis=1)
            predicted_sentence = ""
            for idx in kn_sentence_predicted:
                if idx == future_to_index[END_TOKEN]:
                    break
                predicted_sentence += index_to_Future[idx.item()]
            print(f"Future Prediction : {predicted_sentence}")

            transformer.eval()
            kn_sentence = ("",)
            # eng_sentence = ("should we go to the mall?",)#405 763543
            eng_sentence = ("405 763543",)
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                    eng_sentence, kn_sentence, max_sequence_length)
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
    # Save a checkpoint at every N epochs
    if epoch % save_every_n_epochs == 0:
        save_checkpoint(transformer, optim, epoch, epoch_loss / len(train_loader), checkpoint_dir)
# %% Inference
transformer.eval()


def translate(eng_sentence, transformer, max_sequence_length, index_to_Future, future_to_index, temperature=1.0,
              top_k=5):
    eng_sentence = (eng_sentence,)
    kn_sentence = ("",)
    for word_counter in range(max_sequence_length):
        # Create masks for the transformer model
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            eng_sentence, kn_sentence, max_sequence_length)

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
            eng_sentence, kn_sentence, max_sequence_length)
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
