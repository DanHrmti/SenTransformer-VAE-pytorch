"""
Sentence Transformer-VAE
Authors: Dan Haramati, Nofit Segal
"""

# imports
# general
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import DataLoader
import torchtext
import torchtext.legacy.data as data
import torchtext.legacy.datasets as datasets

"""
Models
"""

# Building Blocks
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Word2SentenceEmbedding(nn.Module):
    def __init__(self, hdim):
        super(Word2SentenceEmbedding, self).__init__()
        self.dense = nn.Linear(hdim, hdim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # take the hidden state corresponding to <sos> token
        first_token_tensor = hidden_states[0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, ntokens, e_dim=200, z_dim=32, nheads=4, nTlayers=4, ff_dim=400, pad_idx=1):
        super(Encoder, self).__init__()
        self.e_dim = e_dim

        self.embedding = nn.Embedding(ntokens, e_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(e_dim)
        encoder_layers = TransformerEncoderLayer(d_model=e_dim, nhead=nheads, dim_feedforward=ff_dim, dropout=0.2)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=nTlayers)
        self.word2sen_hidden = Word2SentenceEmbedding(hdim=e_dim)
        self.hid2latparams = nn.Linear(e_dim, 2 * z_dim)

    def forward(self, sentences, pad_mask):
        embedded = self.embedding(sentences) * math.sqrt(self.e_dim)
        embedded = self.pos_encoding(embedded)
        hidden = self.transformer_encoder(embedded, src_key_padding_mask=pad_mask)
        hidden = self.word2sen_hidden(hidden)
        y = self.hid2latparams(hidden)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar


# Transformer Decoder
class Decoder(nn.Module):
    def __init__(self, ntokens, e_dim=256, z_dim=32, nheads=4, nTlayers=4, ff_dim=1024, pad_idx=1):
        super(Decoder, self).__init__()
        self.e_dim = e_dim

        self.embedding = nn.Embedding(ntokens, e_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(e_dim)
        self.lat2hid = nn.Linear(z_dim, e_dim)
        decoder_layers = TransformerDecoderLayer(d_model=e_dim, nhead=nheads, dim_feedforward=ff_dim, dropout=0.2)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layers, num_layers=nTlayers)
        self.hid2logits = nn.Linear(e_dim, ntokens)

    def forward(self, z, sentences, tgt_mask, tgt_pad_mask):
        memories = self.lat2hid(z)
        memories = memories.unsqueeze(0)
        embedded_targets = self.embedding(sentences) * math.sqrt(self.e_dim)
        embedded_targets = self.pos_encoding(embedded_targets)

        hidden = self.transformer_decoder(embedded_targets, memories, tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_pad_mask)
        logits = self.hid2logits(hidden)
        return logits


# TransformerVAE
class TransformerVAE(nn.Module):
    def __init__(self, ntokens, e_dim=256, z_dim=32, nheads=4, ff_dim=1024, nTElayers=4, nTDlayers=4, pad_idx=1):
        super(TransformerVAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(ntokens, e_dim, z_dim, nheads, nTElayers, ff_dim, pad_idx)
        self.decoder = Decoder(ntokens, e_dim, z_dim, nheads, nTDlayers, ff_dim, pad_idx)

    def forward(self, sentences, pad_mask, tgt_mask):
        mu, logvar = self.encoder(sentences, pad_mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, sentences, tgt_mask, pad_mask)
        return mu, logvar, logits

    def reparameterize(self, mu, logvar):
        """
        This function applies the reparameterization trick:
        z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
        :param mu: mean of x
        :param logvar: log variance of x
        :return z: the sampled latent variable
        """
        device = mu.device
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

    def encode(self, sentence, pad_token=1):
        sentence = sentence.to(device)
        pad_mask = (sentence == pad_token).transpose(0, 1)
        mu, logvar = self.encoder(sentence, pad_mask)
        z = self.reparameterize(mu, logvar)
        return z

    def generate(self, z=None, device="cpu", max_sen_len=50, sos_idx=2, eos_idx=3, policy='greedy', k=10):
        if z is None:  # sample z ~ N(0,I)
            z = torch.randn(self.z_dim).unsqueeze(0).to(device)

        # first target is <sos>
        sen = torch.ones(1, 1).fill_(sos_idx).type(torch.long).to(device)
        for i in range(max_sen_len - 1):
            # create mask
            mask = (torch.triu(torch.ones((len(sen), len(sen)), device=device)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            # decode
            out = self.decoder(z, sen, tgt_mask=mask, tgt_pad_mask=None)
            next_word_prob = out[-1].squeeze()
            if policy == 'randTopK':  # random top k
                _, next_word = torch.topk(next_word_prob, k)
                idx = np.random.randint(0, k, size=1)
                next_word = next_word[idx].item()
            else:  # 'greedy'
                next_word = torch.argmax(next_word_prob)
                next_word = next_word.item()

            # add decoded word to sentence
            sen = torch.cat([sen, torch.ones(1, 1).fill_(next_word).type(torch.long).to(device)], dim=0)
            if next_word == eos_idx:
                break

        return sen

    def reconstruct(self, sen_orig, device="cpu", max_sen_len=50, pad_idx=1, sos_idx=2, eos_idx=3, policy='greedy',
                    k=4):
        # encode original sentence to latent space representation
        z = self.encode(sen_orig, pad_idx)
        # decode from latent space
        sen_rec = self.generate(z, device, max_sen_len, sos_idx, eos_idx, policy, k)
        return sen_rec

    def interpolate(self, sen_1=None, sen_2=None, intervals=10, device="cpu", max_sen_len=50, pad_idx=1, sos_idx=2,
                    eos_idx=3, policy='greedy', k=4):
        if sen_1 is not None:  # encode
            z_1 = self.encode(sen_1, pad_idx)
        else:  # sample z ~ N(0,I)
            z_1 = torch.randn(self.z_dim).unsqueeze(0).to(device)

        if sen_2 is not None:  # encode
            z_2 = self.encode(sen_2, pad_idx)
        else:  # sample z ~ N(0,I)
            z_2 = torch.randn(self.z_dim).unsqueeze(0).to(device)

        sentences = []
        for i in range(intervals + 1):
            t = i / intervals
            z = z_1 * (1 - t) + z_2 * t
            sen = self.generate(z, device, max_sen_len, sos_idx, eos_idx, policy, k)
            sentences.append(sen.squeeze())

        return sentences


"""
Helpers
"""


def data2sentences(data, name='PTB'):
    """
    Divides raw text into sentences (each text requires different parsing)
    :param data: raw text
    :param name: name of text dataset
    :return: sentence_data: list of sentences in raw text
    """
    sentence_data = []
    cur_sentence = []
    text = data.examples[0].text

    if name == 'PTB':
        for i, word in enumerate(text):
            if word == '<eos>':
                sentence_data.append(cur_sentence)
                cur_sentence = []
            else:
                cur_sentence.append(word)

    if name == 'WikiText2':
        for i, word in enumerate(text):
            if ((word == '.' and (len(text[i + 1]) != 1 or text[i + 1] in ['a', 'i'])
                 and (len(text[i - 1]) != 1 or text[i - 1] in [')', ']', '}']))
                    and (text[i - 1] not in ['mr', 'mrs', 'dr', 'jr', 'prof', 'op'])
                    or word in ['<eos>', '?', '!']):
                sentence_data.append(cur_sentence)
                cur_sentence = []
            elif word != "=":
                cur_sentence.append(word)

    return sentence_data


def preprocess_text(name, dataset, textField, max_len=50):
    """
    Divides raw text into padded and tokenized sentences filtered by max_len
    Adds <sos> and <eos> tokens to each sentence
    S - max sequence length
    N - number of sentences in dataset that are <= S
    :param name:   name of text dataset
    :param dataset: raw text
    :param textField: data text_field containing vocabulary and methods
    :param max_len:   maximum sentence length (not including <sos> and <eos> tokens)
    :return: sentences: padded and tokenized sentences, tensor(S,N)
    """
    sentences_raw = data2sentences(dataset, name)

    sentences_filtered = []
    for i in range(len(sentences_raw)):
        if 3 <= len(sentences_raw[i]) <= max_len:
            sentences_filtered.append(sentences_raw[i])

    sentences = textField.pad(sentences_filtered)
    sentences = textField.numericalize(sentences)
    return sentences[:, 1:]


def create_masks(batch, pad_token):
    """
    Create padding and square subsequent masks
    S - max sequence length
    B - batch size
    :param batch: batch of sentences to calculate masks for tensor(S,B)
    :param pad_token: value of pad_token in the vocabulary definition
    :return: pad_mask: padding mask, tensor(S,B)
             mask: square subsequent mask, tensor(S,S)
    """
    pad_mask = (batch == pad_token).transpose(0, 1)

    seq_len = batch.shape[0]
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=batch.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return pad_mask, mask


def calc_kl(mu, logvar, reduce='mean'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param reduce: type of reduce: 'none', 'sum', 'mean
    :return: kl: kl-divergence
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def cyc_beta_scheduler(epochs=20, warmup_epochs=4, beta_min=0, beta_max=0.03, period=8, ratio=0.75):
    beta_warmup = np.ones(warmup_epochs) * beta_min
    beta_cyc = np.ones(epochs - warmup_epochs) * beta_max
    n_cycle = int(np.floor((epochs - warmup_epochs)/period))
    step = (beta_max - beta_min)/(period * ratio)
    for c in range(n_cycle):
        curr_beta, i = beta_min, 0
        while curr_beta <= beta_max and (int(i + c*period) < epochs - warmup_epochs):
              beta_cyc[int(i + c*period)] = curr_beta
              curr_beta += step
              i += 1
    beta = np.concatenate((beta_warmup, beta_cyc), axis=0)
    return beta


"""
Train Function
"""

def train_TransformerVAE(dataset='PTB', e_dim=512, nheads=4, nTElayers=4, nTDlayers=4, z_dim=32,
                         num_epochs=20, batch_size=32, optim='SGDwM', lr=0.1,
                         beta_sch='cyclic', beta_min=0.05, beta_max=0.04, beta_warmup=4, beta_period=8,
                         save_interval=5, device=torch.device("cpu"), seed=-1):

    """
    :param dataset: dataset to train on: ['PTB', 'WikiText2']
    :param e_dim: word embedding dimension
    :param nheads: number of attention heads in transformer encoder/decoder blocks
    :param nTElayers: number of transformer encoder layers
    :param nTDlayers: number of transformer decoder layers
    :param z_dim: latent dimension
    :param num_epochs: total number of epochs to run
    :param batch_size: batch size
    :param optim: optimizer for training: ['SGDwM', 'Adam', 'SGD']
    :param lr: learning rate
    :param beta_sch: beta scheduler: ['cyclic', 'anneal', 'constant']
    :param beta_min: minimum value of beta in scheduler
    :param beta_max: maximum value of beta in scheduler
    :param beta_warmup: number of warmup epochs beta will receive beta_min value
    :param beta_period: number of epochs in a period of the cyclic beta scheduler
    :param save_interval: epochs between checkpoint saving
    :param seed: seed
    :param device: device to run calculation on - torch.device('cuda:x') or torch.device('cpu')
    :return:
    """
    if seed != -1:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print(f"random seed: {seed}\n")

    # ================ Prepare Data ================

    # Initialize tokenizer and text field
    tokenizer = data.utils.get_tokenizer("basic_english")
    textField = data.Field(sequential=True, init_token='<sos>', eos_token='<eos>', lower=True, tokenize=tokenizer)

    # Load and split PennTreebank dataset
    if dataset == 'PTB':
        train_set, valid_set, test_set = datasets.PennTreebank.splits(textField)
    elif dataset == 'WikiText2':
        train_set, valid_set, test_set = datasets.WikiText2.splits(textField)
    else:
        raise NotImplementedError("dataset is not supported")
    print(f'Chosen dataset: {dataset}')

    # Build a vocabulary
    textField.build_vocab(train_set)
    pad_idx = textField.vocab.stoi['<pad>']
    len_vocab = len(textField.vocab)
    print(f'Size of vocabulary: {len_vocab}')

    # Pre-process data into tokenized padded sentences
    max_sen_len = 45
    sentence_data = preprocess_text(dataset, train_set, textField, max_len=max_sen_len)
    num_sentences = sentence_data.shape[1]
    print(f'\nNumber of sentences in train set: {num_sentences}')

    sentence_data_validation = preprocess_text(dataset, valid_set, textField, max_len=max_sen_len)
    num_validation_sentences = sentence_data_validation.shape[1]
    print(f'\nNumber of sentences in validation set: {num_validation_sentences}')

    sentence_data_test = preprocess_text(dataset, test_set, textField, max_len=max_sen_len)
    num_test_sentences = sentence_data_test.shape[1]
    print(f'\nNumber of sentences in test set: {num_test_sentences}\n\n')

    # Print sentence samples from dataset
    # random_sample_indices = np.random.randint(0, sentence_data.shape[1], size=4)
    # for i, s in enumerate(random_sample_indices):
    #     eos_idx = max_sen_len + 2
    #     for idx, token in enumerate(sentence_data[:, s]):
    #         if token == textField.vocab.stoi["<eos>"]:
    #             eos_idx = idx
    #             break
    #     sentence = " ".join([textField.vocab.itos[t] for t in sentence_data[1:eos_idx, s]])
    #     print(f"sample #{s}: {sentence}\n")

    # ================ Model ================

    ntokens = len_vocab
    ff_dim = 4 * e_dim
    model = TransformerVAE(ntokens, e_dim, z_dim, nheads, ff_dim, nTElayers, nTDlayers, pad_idx).to(device)

    if optim == 'SGDwM':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, ], gamma=0.1)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, ], gamma=0.1)
    else:  # 'SGD'
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, ], gamma=0.1)

    if beta_sch == 'cyclic':
        beta_scheduler = cyc_beta_scheduler(epochs=num_epochs, warmup_epochs=beta_warmup, beta_min=beta_min, beta_max=beta_max, period=beta_period, ratio=0.75)
    elif beta_sch == 'anneal':
        beta_scheduler = cyc_beta_scheduler(epochs=num_epochs, warmup_epochs=beta_warmup, beta_min=beta_min, beta_max=beta_max, period=num_epochs-beta_warmup, ratio=0.75)
    else:  # "const"
        beta_scheduler = np.concatenate((beta_min * np.ones(beta_warmup), beta_max * np.ones(num_epochs-beta_warmup)))
    beta = beta_scheduler[0]

    rec_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # print(model)

    # Calculate number of parameters in model
    # dummy_model = TransformerVAE(ntokens, e_dim, z_dim, nheads, ff_dim, nTElayers, nTDlayers, pad_idx)
    # num_trainable_params = sum([p.numel() for p in dummy_model.parameters() if p.requires_grad])
    # print("Total number of parameters: ", num_trainable_params)

    # ================ Training ================

    # Initialize Stats
    rec_loss_log = []
    kl_loss_log = []
    bVAE_loss_log = []  # rec_loss + beta * kl_loss
    log_interval = math.floor((num_sentences / batch_size) / 5)

    # Wrap datasets with DataLoader
    sentences = sentence_data.transpose(0, 1)
    batch_loader = DataLoader(sentences, batch_size=batch_size, shuffle=True)
    sentences_validation = sentence_data_validation.transpose(0, 1)
    validation_batch_loader = DataLoader(sentences_validation, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_rec_loss = 0.
        total_kl_loss = 0.
        total_bVAE_loss = 0.
        total_rec_loss_valid = 0.
        total_kl_loss_valid = 0.
        start_time = time.time()

        for i, batch in enumerate(batch_loader):
            batch = batch.transpose(0, 1).to(device)
            pad_mask, tgt_mask = create_masks(batch, pad_token=pad_idx)
            # forward pass
            mu, logvar, logits = model(batch, pad_mask, tgt_mask)
            # loss calculation
            rec_loss = rec_criterion(logits[:-1, :, :].view(-1, ntokens), batch[1:, :].flatten())
            kl_loss = calc_kl(mu, logvar)
            bVAE_loss = rec_loss + beta * kl_loss
            # backward pass
            optimizer.zero_grad()
            bVAE_loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # update parameters
            optimizer.step()

            # gather and print stats
            total_rec_loss += rec_loss.item()
            total_kl_loss += kl_loss.item()
            total_bVAE_loss += bVAE_loss.item()

            if epoch == 1 and i == 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | beta {:02.4f} | ms/batch {:5.2f} | rec_loss {:5.4f} | kl_loss {:5.4f} | bVAE_loss {:5.4f} |'.format(
                        epoch, i, len(batch_loader), scheduler.get_last_lr()[0], beta, elapsed * 1000 / log_interval,
                        rec_loss, kl_loss, bVAE_loss))

            if i % log_interval == 0 and i > 0:
                cur_rec_loss = total_rec_loss / log_interval
                cur_kl_loss = total_kl_loss / log_interval
                cur_bVAE_loss = total_bVAE_loss / log_interval

                rec_loss_log.append(cur_rec_loss)
                kl_loss_log.append(cur_kl_loss)
                bVAE_loss_log.append(cur_bVAE_loss)

                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | beta {:02.4f} | ms/batch {:5.2f} | rec_loss {:5.4f} | kl_loss {:5.4f} | bVAE_loss {:5.4f} |'.format(
                        epoch, i, len(batch_loader), scheduler.get_last_lr()[0], beta, elapsed * 1000 / log_interval,
                        cur_rec_loss, cur_kl_loss, cur_bVAE_loss))

                total_rec_loss = 0.
                total_kl_loss = 0.
                total_bVAE_loss = 0.
                start_time = time.time()

        # update learning rate
        scheduler.step()
        # update beta
        if epoch < num_epochs:
            beta = beta_scheduler[epoch]

        # evaluate model on validation set
        if epoch % save_interval == 0 or epoch == num_epochs:
            model.eval()
            print("\nEvaluating model on validation set...")
            for i, batch in enumerate(validation_batch_loader):
                batch = batch.transpose(0, 1).to(device)
                pad_mask, tgt_mask = create_masks(batch, pad_token=pad_idx)
                # forward pass
                mu, logvar, logits = model(batch, pad_mask, tgt_mask)
                # loss calculation
                rec_loss_valid = rec_criterion(logits[:-1, :, :].view(-1, ntokens), batch[1:, :].flatten())
                kl_loss_valid = calc_kl(mu, logvar)
                total_rec_loss_valid += rec_loss_valid.item()
                total_kl_loss_valid += kl_loss_valid.item()

            mean_rec_loss_valid = total_rec_loss_valid / len(validation_batch_loader)
            mean_kl_loss_valid = total_kl_loss_valid / len(validation_batch_loader)
            perplexity = math.exp(mean_rec_loss_valid)
            print('| validation_rec_loss {:5.4f} | validation_kl_loss {:5.4f} | validation_perplexity {:5.4f} |'.format(
                    mean_rec_loss_valid, mean_kl_loss_valid, perplexity))

            total_rec_loss_valid = 0.
            total_kl_loss_valid = 0.
            model.train()

        # save model
        print("\n")
        if epoch % save_interval == 0 or epoch == num_epochs:
            print('Saving model ...\n')
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            path = './checkpoints/TransformerVAE_{}_epoch_{:3d}_rec_{:5.4f}_kl_{:5.4f}_valid_perp_{:5.2f}.pth'.format(dataset, epoch, cur_rec_loss, cur_kl_loss, perplexity)
            torch.save(model.state_dict(), path)

    print('\nFinished training')

    # Plot training loss curves
    fig = plt.figure(figsize=(17, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(np.arange(len(rec_loss_log)), rec_loss_log)
    ax1.set_title('Reconstruction Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel(f'Log Interval ({log_interval} batches)')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(np.arange(len(kl_loss_log)), kl_loss_log)
    ax2.set_title('KL Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel(f'Log Interval ({log_interval} batches)')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(np.arange(len(bVAE_loss_log)), bVAE_loss_log)
    ax3.set_title('beta-VAE Loss')
    ax3.set_ylabel('Loss')
    ax3.set_xlabel(f'Log Interval ({log_interval} batches)')

    plt.savefig('./TransformerVAE_loss_curves.jpg')


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    try:
        train_TransformerVAE(dataset='PTB', e_dim=256, nheads=4, nTElayers=4, nTDlayers=4, z_dim=32,
                             num_epochs=40, batch_size=32, optimizer='SGDwM', lr=0.1,
                             beta_sch='cyclic', beta_min=0, beta_max=0.03,
                             save_interval=4, device=torch.device("cpu"), seed=-1)
    except SystemError:
        print("Error, probably loss is NaN, try again...")
