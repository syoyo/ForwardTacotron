from pathlib import Path
from typing import Union, List, Optional

import pysnooper

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F

from models.tacotron import CBHG


class LengthRegulator(nn.Module):

    def __init__(self):
        super().__init__()

    #def forward(self, x: torch.Tensor, dur):
    #    output : List[torch.Tensor] = []
    #    for x_i, dur_i in zip(x, dur):
    #        expanded = self.expand(x_i, dur_i)
    #        output.append(expanded)
    #    output = self.pad(output)
    #    return output

    #def expand(self, x: torch.Tensor, dur):
    #    output : List[torch.Tensor] = []
    #    for i, frame in enumerate(x):
    #        print("dur[i] = ", dur[i])
    #        # TODO(syoyo): Invalid input sequence may generate negative value. Remove assertion in production.
    #        assert dur[i] > 0.0
    #        expanded_len = int(dur[i] + 0.5)
    #        print("expanded_len = ", expanded_len)
    #        print("frame.dim = ", frame.shape, expanded_len)
    #        expanded = frame.expand(expanded_len, -1)
    #        print("expanded.dim = ", expanded.shape)
    #        output.append(expanded)
    #    output = torch.cat(output, 0)
    #    return output

    #def pad(self, x: List[torch.Tensor]):
    #    output : List[torch.Tensor] = []
    #    max_len = max([x[i].size(0) for i in range(len(x))])
    #    for i, seq in enumerate(x):
    #        padded = F.pad(seq, [0, 0, 0, max_len - seq.size(0)], 'constant', 0.0)
    #        output.append(padded)
    #    output = torch.stack(output)
    #    return output

    def forward(self, x, dur):
        return self.expand(x, dur)
     
    @staticmethod
    def build_index(duration, x):
        #print('dur', type(duration))
        #print('x', type(x))
        duration[duration<0]=0
        tot_duration = duration.cumsum(1).detach().cpu().numpy().astype('int')
        max_duration = int(tot_duration.max().item())
        index = np.zeros([x.shape[0], max_duration, x.shape[2]], dtype='long')

        for i in range(tot_duration.shape[0]):
            pos = 0
            for j in range(tot_duration.shape[1]):
                pos1 = tot_duration[i, j]
                index[i, pos:pos1, :] = j
                pos = pos1
            index[i, pos:, :] = j
        return torch.LongTensor(index).to(duration.device)

    def expand(self, x, dur):
        idx = self.build_index(dur, x)
        y = torch.gather(x, 1, idx)
        return y


class DurationPredictor(nn.Module):

    def __init__(self, in_dims, conv_dims=256, rnn_dims=64):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            BatchNormConv(in_dims, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, 1)

    #def forward(self, x, alpha=1.0):
    def forward(self, x):
        alpha = 1.0
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=0.1, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x / alpha


class BatchNormConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, activation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        #if torch.jit.is_scripting():
        #    print("bora")
        #else:
        #    print('activation', self.activation)
        #if self.activation:
        #    x = self.activation(x)

        # activation is always relu, so hardcode it.
        x = torch.relu(x)
        x = self.bnorm(x)
        return x


class ForwardTacotron(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_chars,
                 durpred_conv_dims,
                 durpred_rnn_dims,
                 rnn_dim,
                 prenet_k,
                 prenet_dims,
                 postnet_k,
                 postnet_dims,
                 highways,
                 dropout,
                 n_mels):

        super().__init__()
        self.rnn_dim = rnn_dim
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
        self.dur_pred = DurationPredictor(embed_dims,
                                          conv_dims=durpred_conv_dims,
                                          rnn_dims=durpred_rnn_dims)
        self.prenet = CBHG(K=prenet_k,
                           in_channels=embed_dims,
                           channels=prenet_dims,
                           proj_channels=[prenet_dims, embed_dims],
                           num_highways=highways)
        self.lstm = nn.LSTM(2 * prenet_dims,
                            rnn_dim,
                            batch_first=True,
                            bidirectional=True)
        self.lin = torch.nn.Linear(2 * rnn_dim, n_mels)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.postnet = CBHG(K=postnet_k,
                            in_channels=n_mels,
                            channels=postnet_dims,
                            proj_channels=[postnet_dims, n_mels],
                            num_highways=highways)
        self.dropout = dropout
        self.post_proj = nn.Linear(2 * postnet_dims, n_mels, bias=False)

    #def forward(self, x, mel, dur):
    #    #if not torch.jit.is_scripting():
    #    #    raise "error"

    #    #self.train()
    #    self.step += 1

    #    x = self.embedding(x)
    #    dur_hat = self.dur_pred(x)
    #    dur_hat = dur_hat.squeeze()

    #    x = x.transpose(1, 2)
    #    x = self.prenet(x)
    #    x = self.lr(x, dur)
    #    x, _ = self.lstm(x)
    #    x = F.dropout(x,
    #                  p=self.dropout,
    #                  training=self.training)
    #    x = self.lin(x)
    #    x = x.transpose(1, 2)

    #    x_post = self.postnet(x)
    #    x_post = self.post_proj(x_post)
    #    x_post = x_post.transpose(1, 2)

    #    x_post = self.pad(x_post, mel.size(2))
    #    x = self.pad(x, mel.size(2))
    #    return x, x_post, dur_hat

    def forward(self, x : torch.Tensor):
        # x : Tensor with long type, [batch, N]
        #device = torch.device('cpu')
        #x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        print("fwd x", x.dtype, x.shape)
        x = self.embedding(x)
        dur = self.dur_pred(x)

        x = x.transpose(1, 2)
        x = self.prenet(x)
        x = self.lr(x, dur)
        x, _ = self.lstm(x)
        x = F.dropout(x,
                      p=self.dropout,
                      training=self.training)
        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x_post = x_post.squeeze()
        #x_post = x_post.cpu().data.numpy()
        return x_post

    @torch.jit.unused
    def generate(self, x, alpha=1.0):
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        print("fwd x", x.dtype, x.shape)

        x = self.embedding(x)
        dur = self.dur_pred(x) #, alpha=alpha)
        dur = dur.squeeze(2)

        x = x.transpose(1, 2)
        x = self.prenet(x)
        x = self.lr(x, dur)
        x, _ = self.lstm(x)
        x = F.dropout(x,
                      p=self.dropout,
                      training=self.training)
        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x_post = x_post.squeeze()
        x_post = x_post.cpu().data.numpy()
        print(x_post.dtype, x_post.shape)
        return x_post

    def pad(self, x, max_len: int):
        x = x[:, :, :max_len]
        sz: int = max_len - x.size(2)
        x = F.pad(x, [0, sz, 0, 0], 'constant', 0.0)
        return x

    def get_step(self):
        return self.step.data.item()

    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict, strict=False)

    def save(self, path: Union[str, Path]):
        # No optimizer argument because saving a model should not include data
        # only relevant in the training process - it should only be properties
        # of the model itself. Let caller take care of saving optimzier state.
        torch.save(self.state_dict(), path)

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

