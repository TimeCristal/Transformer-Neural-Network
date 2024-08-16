import os
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Subset, DataLoader
from transformer import Transformer  # this is the transformer.py file
from utils import TextDataset, create_masks, load_checkpoint, save_checkpoint
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch import nn
#%%
# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs/transformer_experiment")

# Define a directory to save the checkpoints
checkpoint_dir = "D:\\checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

# Number of folds for cross-validation
k_folds = 5
#========================
# Transformer params
# checkPoint_id = 400
save_every_n_epochs = 20
num_epochs = 100

d_model = 512 * 2
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 4
max_sequence_length = 40
kn_vocab_size = len(Future_vocabulary)

initial_lr = 1e-4
final_lr = 1e-6
#########################

# Initialize KFold
kf = KFold(n_splits=k_folds, shuffle=True)

# Load data into a dataset
dataset = TextDataset(Present_sentences, Future_sentences)

criterion = nn.CrossEntropyLoss(ignore_index=future_to_index[PADDING_TOKEN],
                                reduction='none')
# K-Fold Cross-Validation Loop
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'FOLD {fold+1}/{k_folds}')
    print('--------------------------------')

    # Split dataset into train and validation subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Re-initialize the model, optimizer, and scheduler for each fold
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
                              PADDING_TOKEN).to(device)

    optim = torch.optim.Adam(transformer.parameters(), lr=initial_lr)
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=5, T_mult=2, eta_min=final_lr)

    # Try to load a checkpoint if resuming
    start_epoch, best_loss = load_checkpoint(transformer, optim, checkpoint_dir,
                                             filename=f"checkpoint_fold{fold}.pth")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch}")
        iterator = iter(train_loader)
        epoch_loss = 0

        # Training loop
        transformer.train()
        for batch_num, batch in enumerate(iterator):
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
            valid_indices = torch.where(labels.view(-1) == future_to_index[PADDING_TOKEN], False, True)
            loss = loss.sum() / valid_indices.sum()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

            # Log loss to TensorBoard
            writer.add_scalar(f'Loss/Train/Fold_{fold+1}', loss.item(), epoch * len(train_loader) + batch_num)

        # Validation loop
        transformer.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                eng_batch, kn_batch = val_batch
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
                    eng_batch,
                    kn_batch,
                    max_sequence_length)
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
                valid_indices = torch.where(labels.view(-1) == future_to_index[PADDING_TOKEN], False, True)
                loss = loss.sum() / valid_indices.sum()
                val_loss += loss.item()

        # Log validation loss
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar(f'Loss/Validation/Fold_{fold+1}', avg_val_loss, epoch)

        # Update learning rate scheduler
        scheduler.step()

        # Save a checkpoint
        if epoch % save_every_n_epochs == 0:
            save_checkpoint(transformer, optim, epoch, avg_val_loss, checkpoint_dir,
                            filename=f"checkpoint_fold{fold}_epoch{epoch}.pth")

    print(f'Finished Fold {fold+1}/{k_folds}')
    print('--------------------------------')

# Close TensorBoard writer
writer.close()