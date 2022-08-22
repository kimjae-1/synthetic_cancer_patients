# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable

from basic import PositionwiseFeedForward, PositionalEncoding, TimeEncoding, max_pooling, mean_pooling

from fastNLP import DataSet, DataSetIter, RandomSampler, SequentialSampler, seq_len_to_mask

import random
import time
import math, copy
import pandas as pd


class Embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(Embedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.pos = TimeEncoding(hidden_dim)

    def forward(self, x, times):
        x = self.fc(x)
        return self.dropout(x + self.pos(times))


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, layers, dropout):
        super(Encoder, self).__init__()  ##python 2 style?
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.embed = Embedding(input_dim, 2 * hidden_dim, dropout)
        self.rnn = nn.GRU(2 * hidden_dim, 2 * hidden_dim, layers, batch_first=True, bidirectional=False,
                          dropout=dropout)

        self.fc = nn.Linear(2 * hidden_dim * 3, 2 * hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim * layers, 2 * hidden_dim * layers)
        self.final = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        self.poisson = nn.Linear(8*hidden_dim, 4*hidden_dim)
        # self.halved_size = 2 * hidden_dim
        # self.size_matcher = nn.Linear(2*hidden_dim, 4*hidden_dim)

    def forward(self, statics, dynamics, priv, nex, mask, times, seq_len):
        bs, max_len, _ = dynamics.size()
        x = statics.unsqueeze(1).expand(-1, max_len, -1)
        x = torch.cat([x, dynamics, priv, mask], dim=-1)
        x = self.embed(x, times)

        packed = nn.utils.rnn.pack_padded_sequence(x, seq_len.to('cpu'), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        h = h.view(self.layers, -1, bs, self.hidden_dim)
        h1, _ = max_pooling(out, seq_len)
        h2 = mean_pooling(out, seq_len)
        h3 = h[-1].view(bs, -1)
        glob = torch.cat([h1, h2, h3], dim=-1)
        glob = self.final(self.fc(self.drop(glob)))

        lasth = h.view(-1, bs, self.hidden_dim)
        lasth = lasth.permute(1, 0, 2).contiguous().view(bs, -1)
        lasth = self.final(self.fc1(self.drop(lasth)))

        hidden = torch.cat([glob, lasth], dim=-1)

        z_poisson = torch.poisson(self.poisson(hidden))

        mu, logvar = hidden[:, :4 * self.hidden_dim], hidden[:, 4 * self.hidden_dim:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + (std * eps) + z_poisson

        return [z, mu, logvar]


def apply_activation(processors, x):
    data = []
    st = 0
    for model in processors.models:
        if model.name == processors.use_pri:
            continue
        ed = model.tgt_len + st
        if model.which == 'categorical':
            if not model.missing or processors.use_pri:
                data.append(torch.softmax(x[:, st:ed], dim=-1))
            else:
                data.append(torch.softmax(x[:, st:ed - 1], dim=-1))
                data.append(torch.sigmoid(x[:, ed - 1:ed]))
            st = ed
        else:
            data.append(torch.sigmoid(x[:, st:ed]))
            st = ed
    assert ed == x.size(1)
    return torch.cat(data, dim=-1)


def pad_zero(x):
    input_x = torch.zeros_like(x[:, 0:1, :])
    input_x = torch.cat([input_x, x[:, :-1, :]], dim=1)
    return input_x


class Decoder(nn.Module):
    def __init__(self, processors, hidden_dim, layers, dropout):
        super(Decoder, self).__init__()
        self.s_P, self.d_P = processors
        self.hidden_dim = hidden_dim
        statics_dim, dynamics_dim = self.s_P.tgt_dim, self.d_P.tgt_dim
        self.dynamics_dim = dynamics_dim
        self.miss_dim = self.d_P.miss_dim
        self.s_dim = sum([x.tgt_len for x in self.d_P.models if x.missing])
        self.layers = layers
        self.embed = Embedding(dynamics_dim + self.s_dim + statics_dim + self.miss_dim, hidden_dim, dropout)
        self.rnn = nn.GRU(2*hidden_dim, hidden_dim, 1, batch_first=True, dropout=dropout)
        self.miss_rnn = nn.GRU(2*hidden_dim, hidden_dim, layers - 1, batch_first=True, dropout=dropout)

        self.statics_fc = nn.Linear(hidden_dim, statics_dim)
        self.dynamics_fc = nn.Linear(hidden_dim, dynamics_dim)
        self.decay = nn.Linear(self.miss_dim * 2, hidden_dim)

        self.miss_fc = nn.Linear(hidden_dim, self.miss_dim)
        self.time_fc = nn.Linear(hidden_dim, 1)

    def forward(self, embed, sta, dynamics, lag, mask, priv, times, seq_len, forcing=11):
        ## after 0801
        ## embed = hidden(z), mu, logvar

        ##before 0801
        # embed == hidden == z == encoder output
        # embed shape (batch_size, 4*hidden_dim) ; should be (128,512), but halved to (128, 256)
        glob, hidden = embed[0][:, :self.hidden_dim], embed[0][:, self.hidden_dim:]
        mu, logvar = embed[1], embed[2]
        #z_poisson = embed[3]
        # so hidden must be sized ; (batch_size, 3 * hidden_dim)
        statics_x = self.statics_fc(glob)
        gen_sta = apply_activation(self.s_P, statics_x)
        bs, max_len, _ = dynamics.size()  # bs=batch_size
        # as layers == 3,
        #hidden = hidden + z_poisson
        hidden = hidden.view(bs, self.layers, -1).permute(1, 0, 2).contiguous()
        hidden, finh = hidden[:-1], hidden[-1:]

        # force ?
        if forcing >= 1:
            pad_dynamics = pad_zero(dynamics)
            pad_mask = pad_zero(mask)
            pad_times = pad_zero(times)
            pad_priv = pad_zero(priv)
            sta_expand = sta.unsqueeze(1).expand(-1, max_len, -1)
            glob_expand = glob.unsqueeze(1).expand(-1, max_len, -1)
            x = torch.cat([sta_expand, pad_dynamics, pad_priv, pad_mask], dim=-1)
            x = self.embed(x, pad_times)
            packed = nn.utils.rnn.pack_padded_sequence(x, seq_len.to('cpu'), batch_first=True, enforce_sorted=False)

            out, h = self.miss_rnn(packed, hidden)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            gen_times = torch.sigmoid(self.time_fc(out)) + pad_times
            gen_mask = torch.sigmoid(self.miss_fc(out))

            beta = torch.exp(-torch.relu(self.decay(torch.cat([mask, lag], dim=-1))))
            y = beta * out

            y = torch.cat([y, glob_expand], dim=-1)
            packed = nn.utils.rnn.pack_padded_sequence(y, seq_len.to('cpu'), batch_first=True, enforce_sorted=False)
            out1, finh = self.rnn(packed, finh)
            out1, _ = torch.nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)

            dyn = self.dynamics_fc(out1)
            dyn = apply_activation(self.d_P, dyn.view(-1, self.dynamics_dim)).view(bs, -1, self.dynamics_dim)

        else:
            true_sta = sta.unsqueeze(1)
            gsta = gen_sta.detach().unsqueeze(1)
            glob = glob.unsqueeze(1)
            dyn = []
            gen_mask = []
            gen_times = []
            cur_x = torch.zeros((bs, 1, self.dynamics_dim)).to(embed.device)
            gen_p = [torch.zeros((bs, 1, self.s_dim)).to(embed.device)]
            cur_mask = torch.zeros((bs, 1, self.miss_dim)).to(embed.device)
            cur_time = torch.zeros((bs, 1, 1)).to(embed.device)
            thr = torch.Tensor([model.threshold for model in self.d_P.models if model.missing]).to(embed.device)
            thr = thr.view(1, 1, self.miss_dim).expand(bs, -1, -1)

            force = True
            for i in range(max_len):
                force = random.random() < forcing
                if i == 0 or not force:
                    sta = gsta
                    pre_x = cur_x
                    pre_mask = cur_mask.detach()
                    pre_time = cur_time.detach()
                else:
                    sta = true_sta
                    pre_x = dynamics[:, i - 1:i]
                    pre_mask = mask[:, i - 1:i]
                    pre_time = times[:, i - 1:i]

                j = 0
                st = 0
                np = gen_p[-1].detach()
                for model in self.d_P.models:
                    if model.name == self.d_P.use_pri: continue
                    if model.missing:
                        np[:, :, st:st + model.tgt_len] = np[:, :, st:st + model.tgt_len] * (
                                    1 - pre_mask[:, :, j:j + 1]) + pre_x[:, :, st:st + model.tgt_len] * pre_mask[:, :,
                                                                                                        j:j + 1]
                        j += 1
                    st += model.tgt_len
                gen_p.append(np)

                in_x = torch.cat([sta, pre_x, gen_p[i], pre_mask], dim=-1)
                in_x = self.embed(in_x, pre_time)

                out, hidden = self.miss_rnn(in_x, hidden)
                cur_time = torch.sigmoid(self.time_fc(out))

                if i == 0:
                    lg = cur_time.expand(-1, -1, self.miss_dim).detach()
                else:
                    lg = (1 - pre_mask) * lg + cur_time.detach()
                if i > 0:
                    gen_times.append(cur_time + pre_time)
                else:
                    gen_times.append(cur_time)
                cur_mask = torch.sigmoid(self.miss_fc(out))
                gen_mask.append(cur_mask)
                if force:
                    use_mask = mask[:, i:i + 1]
                else:
                    use_mask = cur_mask.detach()

                beta = torch.exp(-torch.relu(self.decay(torch.cat([use_mask, lg], dim=-1))))
                y = torch.cat([out * beta, glob], dim=-1)

                out, finh = self.rnn(y, finh)
                out = self.dynamics_fc(out)
                out = apply_activation(self.d_P, out.squeeze(1)).unsqueeze(1)
                dyn.append(out)

                x = out.detach()
                x = self.d_P.re_transform(x.squeeze(1).cpu().numpy(), use_mask.squeeze(1).cpu().numpy())
                cur_x = torch.FloatTensor(x).to(embed.device).unsqueeze(1)
                cur_time = gen_times[-1].detach()
                cur_mask = (cur_mask > thr).detach().float()

            dyn = torch.cat(dyn, dim=1)
            gen_mask = torch.cat(gen_mask, dim=1)
            gen_times = torch.cat(gen_times, dim=1)

        return gen_sta, dyn, gen_mask, gen_times, mu, logvar

    def generate_statics(self, embed):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        statics = self.statics_fc(glob)
        statics = apply_activation(self.s_P, statics)
        return statics.detach()

    def generate_dynamics(self, embed, sta, max_len):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        glob = glob.unsqueeze(1)
        sta = sta.unsqueeze(1)
        bs = glob.size(0)
        x = torch.zeros((bs, 1, self.dynamics_dim)).to(embed.device)
        priv = [torch.zeros((bs, 1, self.s_dim)).to(embed.device), torch.zeros((bs, 1, self.s_dim)).to(embed.device)]
        mask = torch.zeros((bs, 1, self.miss_dim)).to(embed.device)
        t = torch.zeros((bs, 1, 1)).to(embed.device)
        hidden = hidden.view(bs, self.layers, -1).permute(1, 0, 2).contiguous()
        hidden, regress_hidden = hidden[:-1], hidden[-1:]
        dyn = []
        gen_mask = []
        gen_times = []
        thr = torch.Tensor([model.threshold for model in self.d_P.models if model.missing]).to(embed.device)
        thr = thr.view(1, 1, self.miss_dim).expand(bs, -1, -1)
        for i in range(max_len):
            in_x = torch.cat([sta, x, priv[i], mask], dim=-1)
            in_x = self.embed(in_x, t)
            out, hidden = self.miss_rnn(in_x, hidden)
            cur_times = torch.sigmoid(self.time_fc(out))

            if i == 0:
                lg = cur_times.expand(-1, -1, self.miss_dim)
            else:
                lg = (1 - mask) * lg + cur_times
            if i > 0:
                gen_times.append(cur_times + gen_times[-1])
            else:
                gen_times.append(cur_times)
            mask = torch.sigmoid(self.miss_fc(out))
            gen_mask.append(mask)

            beta = torch.exp(-torch.relu(self.decay(torch.cat([mask, lg], dim=-1))))
            y = torch.cat([out * beta, glob], dim=-1)

            out, regress_hidden = self.rnn(y, regress_hidden)
            out = self.dynamics_fc(out)
            out = apply_activation(self.d_P, out.squeeze(1)).unsqueeze(1)
            dyn.append(out)

            j = 0
            x = out.detach()
            x = self.d_P.re_transform(x.squeeze(1).cpu().numpy(), mask.detach().squeeze(1).cpu().numpy())
            x = torch.FloatTensor(x).to(embed.device).unsqueeze(1)
            mask = (mask > thr).float()
            st = 0
            np = priv[-1].detach()
            for model in self.d_P.models:
                if model.name == self.d_P.use_pri: continue
                if model.missing:
                    # x[:, :, st:st+model.tgt_len] *= mask[:,:,j:j+1]
                    np[:, :, st:st + model.tgt_len] = \
                        np[:, :, st:st + model.tgt_len] * (1 - mask[:, :, j:j + 1]) + x[:, :,
                                                                                      st:st + model.tgt_len] * mask[:,
                                                                                                               :,
                                                                                                               j:j + 1]

                    j += 1
                st += model.tgt_len
            priv.append(np)
            t = gen_times[-1].detach()
            mask = mask.detach()

        dyn = torch.cat(dyn, dim=1)
        gen_mask = torch.cat(gen_mask, dim=1)
        gen_times = torch.cat(gen_times, dim=1)
        return dyn, gen_mask, gen_times


class Autoencoder(nn.Module):
    def __init__(self, processors, hidden_dim, embed_dim, layers, dropout=0.0):
        super(Autoencoder, self).__init__()
        #print(processors[0].tgt_dim, processors[1].tgt_dim, processors[1].miss_dim)  # f'string
        s_dim = sum([x.tgt_len for x in processors[1].models if x.missing])
        self.encoder = Encoder(processors[0].tgt_dim + processors[1].tgt_dim + s_dim + processors[1].miss_dim,
                               hidden_dim, embed_dim, layers, dropout)
        self.decoder = Decoder(processors, hidden_dim, layers, dropout)
        self.decoder.embed = self.encoder.embed

    def forward(self, sta, dyn, lag, mask, priv, nex, times, seq_len, forcing=1):
        hidden = self.encoder(sta, dyn, priv, nex, mask, times, seq_len)
        return self.decoder(hidden, sta, dyn, lag, mask, priv, times, seq_len, forcing=forcing)


class AeGAN:
    def __init__(self, processors, params):
        self.params = params
        if self.params.get("force") is None:
            self.params["force"] = ""
        self.device = params["device"]
        self.logger = params["logger"]
        self.static_processor, self.dynamic_processor = processors

        self.ae = Autoencoder(
            processors, self.params["hidden_dim"], self.params["embed_dim"], self.params["layers"],
            dropout=self.params["dropout"])
        self.ae.to(self.device)

        self.ae_optm = torch.optim.Adam(
            params=self.ae.parameters(),
            lr=self.params['ae_lr'],
            betas=(0.9, 0.999),
            weight_decay=self.params["weight_decay"]
        )

        self.loss_con = nn.MSELoss(reduction='none')
        self.loss_dis = nn.NLLLoss(reduction='none')
        self.loss_mis = nn.BCELoss(reduction='none')

    def load_ae(self, pretrained_dir=None):
        if pretrained_dir is not None:
            path = pretrained_dir
        else:
            path = f'{self.params["root_dir"]}/ae.dat'
        self.logger.info("load: " + path)
        self.ae.load_state_dict(torch.load(path, map_location=self.device))

    def sta_loss(self, data, target):
        loss = 0
        n = len(self.static_processor.models)
        st = 0
        # print(data,target)
        for i, model in enumerate(self.static_processor.models):
            ed = st + model.tgt_len - int(model.missing)
            use = 1
            if model.missing:
                loss += 0.1 * torch.mean(self.loss_mis(data[:, ed], target[:, ed]))
                use = target[:, ed:ed + 1]

            if model.which == "categorical":
                loss += torch.mean(use * self.loss_dis((data[:, st:ed] + 1e-8).log(),
                                                       torch.argmax(target[:, st:ed], dim=-1)).unsqueeze(-1))
            elif model.which == "binary":
                loss += torch.mean(use * self.loss_mis(data[:, st:ed], target[:, st:ed]))
            else:
                loss += torch.mean(use * self.loss_con(data[:, st:ed], target[:, st:ed]))

            st += model.tgt_len
        assert st == target.size(-1)
        return loss / n

    def dyn_loss(self, data, target, seq_len, mask):
        loss = []
        n = len(self.dynamic_processor.models)
        st = 0
        i = 0
        for model in self.dynamic_processor.models:
            if model.name == self.dynamic_processor.use_pri: continue
            ed = st + model.tgt_len
            use = 1
            if model.missing:
                use = mask[:, :, i:i + 1]
                i += 1

            if model.which == "categorical":
                x = (data[:, :, st:ed] + 1e-8).log().transpose(1, 2)
                loss.append(use * self.loss_dis(x, torch.argmax(target[:, :, st:ed], dim=-1)).unsqueeze(-1))
            elif model.which == "binary":
                loss.append(use * self.loss_mis(data[:, :, st:ed], target[:, :, st:ed]))
            else:
                loss.append(use * 10 * self.loss_con(data[:, :, st:ed], target[:, :, st:ed]))
            st += model.tgt_len
        assert i == mask.size(-1)
        loss = torch.cat(loss, dim=-1)
        seq_mask = seq_len_to_mask(seq_len)
        loss = torch.masked_select(loss, seq_mask.unsqueeze(-1))
        return torch.mean(loss)

    def time_loss(self, data, target, seq_len):
        loss = self.loss_con(data, target)
        seq_mask = seq_len_to_mask(seq_len)
        loss = torch.masked_select(loss, seq_mask.unsqueeze(-1))
        return torch.mean(loss)

    def missing_loss(self, data, target, seq_len):
        thr = torch.Tensor([model.threshold for model in self.dynamic_processor.models if model.missing]).to(
            data.device)
        thr = thr.unsqueeze(0).unsqueeze(0)

        scale = thr * target + (1 - thr) * (1 - target)
        loss = self.loss_mis(data, target) * scale
        seq_mask = seq_len_to_mask(seq_len)
        loss = torch.masked_select(loss, seq_mask.unsqueeze(-1))

        mx, _ = max_pooling(data, seq_len)
        gold_mx, _ = torch.max(target, dim=1)
        loss1 = self.loss_mis(mx, gold_mx)
        return torch.mean(loss) + torch.mean(torch.masked_select(loss1, gold_mx == 0))

    def train_ae(self, dataset, i):
        target_batch = DataSetIter(dataset=dataset, batch_size=self.params["ae_batch_size"], sampler=RandomSampler())
        force = 1

        self.ae.train()

        tot_loss = 0
        con_loss = 0
        dis_loss = 0
        miss_loss1 = 0
        miss_loss2 = 0
        tot = 0
        t1 = time.time()

        if self.params["force"] == "linear":
            if i >= epochs / 100 and i < epochs / 2:
                force -= 2 / epochs
            elif i >= epochs / 2:
                force = -1
        elif self.params["force"] == "constant":
            force = 0.5
        else:
            force = 1

        for batch_x, batch_y in target_batch:
            self.ae.zero_grad()
            sta = batch_x["sta"].to(self.device)
            dyn = batch_x["dyn"].to(self.device)
            mask = batch_x["mask"].to(self.device)
            lag = batch_x["lag"].to(self.device)
            priv = batch_x["priv"].to(self.device)
            nex = batch_x["nex"].to(self.device)
            times = batch_x["times"].to(self.device)
            seq_len = batch_x["seq_len"].to(self.device)

            out_sta, out_dyn, missing, gt, mu, logvar = self.ae(sta, dyn, lag, mask, priv, nex, times, seq_len, forcing=force)
            KLD = - 0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar)) / (self.params['ae_batch_size'])
            loss3 = self.missing_loss(missing, mask, seq_len)
            miss_loss1 += loss3.item()
            loss4 = self.time_loss(gt, times, seq_len)
            miss_loss2 += loss4.item()

            loss1 = self.sta_loss(out_sta, sta)
            loss2 = self.dyn_loss(out_dyn, dyn, seq_len, mask)

            sta_num = len(self.static_processor.models)
            dyn_num = len(self.dynamic_processor.models)
            scale1 = sta_num / (sta_num + dyn_num)
            scale2 = dyn_num / (sta_num + dyn_num)
            scale3 = 1.2
            scale4 = 1

            #2, 0,001?
            #1, 0.0001
            #1.5, 1.1 , 0.0001
            loss = (scale1 * loss1) + (scale2 * loss2) + (scale3 * loss3) + (scale4 * loss4) + (0.001 * KLD)
            # loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            self.ae_optm.step()

            tot_loss += loss.item()
            con_loss += loss1.item()
            dis_loss += loss2.item()
            tot += 1

        tot_loss /= tot
        #print(loss1.item(), loss2.item(), loss3.item(), loss4.item())
        return [tot_loss, time.time() - t1, con_loss, dis_loss, miss_loss1, miss_loss2, KLD, tot]

    def eval_ae(self, dataset):
        batch_size = self.params["ae_batch_size"]
        batch = DataSetIter(dataset=dataset, batch_size=batch_size, sampler=SequentialSampler())

        con_loss = 0
        dis_loss = 0
        miss_loss1 = 0
        miss_loss2 = 0
        tot_loss = 0
        tot = 0

        sta_lis = []
        dyn_lis = []
        h = []

        self.ae.eval()
        with torch.no_grad():
            for batch_x, batch_y in batch:
                sta = batch_x["sta"].to(self.device)
                dyn = batch_x["dyn"].to(self.device)
                mask = batch_x["mask"].to(self.device)
                lag = batch_x["lag"].to(self.device)
                priv = batch_x["priv"].to(self.device)
                nex = batch_x["nex"].to(self.device)
                times = batch_x["times"].to(self.device)
                seq_len = batch_x["seq_len"].to(self.device)

                hidden, mu, logvar = self.ae.encoder(sta, dyn, priv, nex, mask, times, seq_len)
                h.append(hidden)
                statics = self.ae.decoder.generate_statics(hidden)
                max_len = dyn.size(1)
                dynamics, missing, gt = self.ae.decoder.generate_dynamics(hidden, statics, max_len)

                sta_num = len(self.static_processor.models)
                dyn_num = len(self.dynamic_processor.models)
                scale1 = sta_num / (sta_num + dyn_num)
                scale2 = dyn_num / (sta_num + dyn_num)
                scale3 = 0.1

                loss1 = self.sta_loss(statics, sta)
                loss2 = self.dyn_loss(dynamics, dyn, seq_len, mask)
                loss3 = self.missing_loss(missing, mask, seq_len)
                loss4 = self.time_loss(gt, times, seq_len)

                loss = scale1 * loss1 + scale2 * (loss2 + loss3) + scale3 * loss4

                tot_loss += loss.item()
                con_loss += loss1.item()
                dis_loss += loss2.item()
                miss_loss1 += loss3.item()
                miss_loss2 += loss4.item()
                tot += 1

                dynamics = dynamics.cpu().numpy()
                missing = missing.cpu().numpy()
                times = times.cpu().numpy()
                df_sta = self.static_processor.inverse_transform(statics.cpu().numpy())
                sta_lis.append(df_sta)
                for i, length in enumerate(seq_len.cpu().numpy()):
                    d = self.dynamic_processor.inverse_transform(dynamics[i, :length], missing[i, :length],
                                                                 times[i, :length])
                    dyn_lis.append(d)

            sta_lis = pd.concat(sta_lis)

        tot_loss /= tot
        h = torch.cat(h, dim=0).cpu().numpy()

        return [tot_loss, 0, con_loss, dis_loss, miss_loss1, miss_loss2, 'KLD', tot, h, sta_lis, dyn_lis]  # 0=time_spaceholder

    def synthesize(self, n, batch_size=500):
        self.ae.decoder.eval()
        sta = []
        seq_len = []
        dyn = []

        def _gen(n):
            with torch.no_grad():
                hidden = torch.randn(n, 4 * self.params['hidden_dim']).to(self.device)
                # hidden = self.ae.decoder(z)
                statics = self.ae.decoder.generate_statics(hidden)
                df_sta = self.static_processor.inverse_transform(statics.cpu().numpy())
                max_len = int(df_sta['seq_len'].max())
                sta.append(df_sta)
                cur_len = df_sta['seq_len'].values.astype(int).tolist()
                dynamics, missing, times = self.ae.decoder.generate_dynamics(hidden, statics, max_len)
                dynamics = dynamics.cpu().numpy()
                missing = missing.cpu().numpy()
                times = times.cpu().numpy()
                for i, length in enumerate(cur_len):
                    d = self.dynamic_processor.inverse_transform(dynamics[i, :length], missing[i, :length],
                                                                 times[i, :length])
                    dyn.append(d)

        tt = n // batch_size
        for i in range(tt):
            _gen(batch_size)
        res = n - tt * batch_size
        if res > 0:
            _gen(res)
        sta = pd.concat(sta)
        assert len(sta) == len(dyn)
        print(sta[0:1])
        print(dyn[0])
        return (sta, dyn)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def mean_pooling(tensor, seq_len, dim=1):
    mask = seq_len_to_mask(seq_len)
    mask = mask.view(mask.size(0), mask.size(1), -1).float()
    return torch.sum(tensor * mask, dim=dim) / seq_len.unsqueeze(-1).float()


def max_pooling(tensor, seq_len, dim=1):
    mask = seq_len_to_mask(seq_len)
    mask = mask.view(mask.size(0), mask.size(1), -1)
    mask = mask.expand(-1, -1, tensor.size(2)).float()
    return torch.max(tensor + mask.le(0.5).float() * -1e9, dim=dim)


class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super(TimeEncoding, self).__init__()
        self.fc = nn.Linear(1, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.fc(x)
        x = torch.cat([x[:, :, 0:1], torch.sin(x[:, :, 1:])], dim=-1)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).unsqueeze(0)
        div_term.require_grad = False
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        bs, lens, _ = x.size()
        x = x.view(-1, 1)
        pe = torch.zeros(x.size(0), self.d_model).to(x.device)
        x = x * 100
        pe[:, 0::2] = torch.sin(x * self.div_term)
        pe[:, 1::2] = torch.cos(x * self.div_term)
        return Variable(pe.view(bs, lens, -1), requires_grad=False)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, input_dim, output_dim, d_ff=None, activation=F.relu, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        if d_ff is None:
            d_ff = output_dim * 4
        self.act = activation
        self.w_1 = nn.Linear(input_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))


def dot_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # print(scores.size(),mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)


class BahdanauAttention(nn.Module):
    def __init__(self, d_k):
        super(BahdanauAttention, self).__init__()
        self.alignment_layer = nn.Linear(d_k, 1, bias=False)

    def forward(self, query, key, value, mask=None):
        query = query.unsqueeze(-2)
        key = key.unsqueeze(-3)
        scores = self.alignment_layer(query + key).squeeze(-1)
        if mask is not None:
            # print(scores.size(),mask.size())
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value)


class SelfAttention(nn.Module):
    def __init__(self, d_model, h=2, dropout=0.1):
        "Take in model size and number of heads."
        super(SelfAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = BahdanauAttention(self.d_k)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attn(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.dropout(self.linears[-1](x))


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)