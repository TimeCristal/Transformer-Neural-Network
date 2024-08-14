import os

import numpy as np
import torch
from torch.utils.data import Dataset


def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True


def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1)  # need to re-add the end token so leaving 1 space


#%% md
# Modify mask such that the padding tokens cannot look ahead.
# In Encoder, tokens before it should be -1e9 while tokens after it should be -inf.
#
#%% md
# Note the target mask starts with 2 rows of non-masked items: https://github.com/SamLynnEvans/Transformer/blob/master/
# Beam.py#L55
NEG_INFINITY = -1e9


def create_masks(eng_batch, kn_batch, max_sequence_length):
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

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFINITY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFINITY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFINITY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


class TextDataset(Dataset):

    def __init__(self, present_sentences, future_sentences):
        self.Present_sentences = present_sentences
        self.Future_sentences = future_sentences

    def __len__(self):
        return len(self.Present_sentences)

    def __getitem__(self, idx):
        return self.Present_sentences[idx], self.Future_sentences[idx]


# # Define a directory to save the checkpoints
# checkpoint_dir = "checkpoints"
# os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename='checkpoint.pth'):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, checkpoint_dir, filename='checkpoint.pth'):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from epoch {epoch}")
        return epoch, loss
    else:
        print("No checkpoint found.")
        return 0, float('inf')  # Start from scratch
#
# # Training loop
# transformer.train()
# transformer.to(device)
# total_loss = 0
#
# # Attempt to load the last checkpoint
# start_epoch, best_loss = load_checkpoint(transformer, optim, checkpoint_dir)
#
# for epoch in range(start_epoch, num_epochs):
#     print(f"Epoch {epoch}")
#     iterator = iter(train_loader)
#     epoch_loss = 0
#
#     for batch_num, batch in enumerate(iterator):
#         transformer.train()
#         eng_batch, kn_batch = batch
#         encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask =
#         create_masks(eng_batch,
#                                                                                                               kn_batch)
#         optim.zero_grad()
#         kn_predictions = transformer(eng_batch,
#                                      kn_batch,
#                                      encoder_self_attention_mask.to(device),
#                                      decoder_self_attention_mask.to(device),
#                                      decoder_cross_attention_mask.to(device),
#                                      enc_start_token=False,
#                                      enc_end_token=False,
#                                      dec_start_token=True,
#                                      dec_end_token=True)
#         labels = transformer.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
#         loss = criteria(
#             kn_predictions.view(-1, kn_vocab_size).to(device),
#             labels.view(-1).to(device)
#         ).to(device)
#         valid_indices = torch.where(labels.view(-1) == Future_to_index[PADDING_TOKEN], False, True)
#         loss = loss.sum() / valid_indices.sum()
#         loss.backward()
#         optim.step()
#         epoch_loss += loss.item()
#
#         # Periodically save checkpoints
#         if batch_num % 100 == 0:
#             print(f"Iteration {batch_num} : {loss.item()}")
#             save_checkpoint(transformer, optim, epoch, loss.item(), checkpoint_dir)
#
#     print(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader)}")
#
#     # Save a checkpoint at the end of each epoch
#     save_checkpoint(transformer, optim, epoch, epoch_loss / len(train_loader), checkpoint_dir)
