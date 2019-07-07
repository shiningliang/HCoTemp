import torch
from torch import nn
from time import time


class COTEMP(nn.Module):
    def __init__(self, u_emb, i_emb, T, NU, NI, NF, output_size, n_hidden, n_layer, dropout, logger):
        super(COTEMP, self).__init__()
        start_t = time()

        self.T = T
        self.n_user = NU
        self.n_item = NI
        self.n_emb = NF
        self.gru_hidden = n_hidden
        self.n_layer = n_layer
        self.user_embedding = nn.Embedding(self.n_user * self.T + 1, self.n_emb, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_item * self.T + 1, self.n_emb, padding_idx=0)

        self.emb_dropout = nn.Dropout(dropout['emb'])
        self.user_encoder = nn.GRU(2 * self.n_emb, n_hidden, n_layer, dropout=dropout['layer'], batch_first=True,
                                   bidirectional=True)
        self.item_encoder = nn.GRU(2 * self.n_emb, n_hidden, n_layer, dropout=dropout['layer'], batch_first=True,
                                   bidirectional=True)
        self.out_fc = nn.Linear(4 * n_hidden, output_size)

        self._init_weights(u_emb, i_emb)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, u_emb, i_emb):
        self.user_embedding.weight.data.copy_(torch.from_numpy(u_emb))
        self.item_embedding.weight.data.copy_(torch.from_numpy(i_emb))

    def forward(self, user_records, item_records, uids, iids):
        batch_size = user_records.shape[0]
        umax_len = user_records.shape[2]
        imax_len = item_records.shape[2]
        urecords = torch.reshape(user_records, (batch_size * self.T, umax_len))
        irecords = torch.reshape(item_records, (batch_size * self.T, imax_len))
        user_masks = torch.where(urecords > 0, torch.full_like(urecords, 1), urecords)
        item_masks = torch.where(irecords > 0, torch.full_like(irecords, 1), irecords)

        urecords_embs = self.item_embedding(urecords)  # user的record是item
        irecords_embs = self.user_embedding(irecords)
        uid_embs = self.user_embedding(uids)  # user每个月的embedding
        # uid_embs = torch.unsqueeze(uid_embs, 1)
        # uid_embs = uid_embs.repeat(1, T, 1)
        iid_embs = self.item_embedding(iids)
        # iid_embs = torch.unsqueeze(1, T, 1)
        # iid_embs = iid_embs.repeat(1, T)

        urecords_embs = torch.sum(urecords_embs, dim=1)
        irecords_embs = torch.sum(irecords_embs, dim=1)
        user_masks = torch.sum(user_masks, dim=-1, keepdim=True)
        item_masks = torch.sum(item_masks, dim=-1, keepdim=True)
        user_masks = user_masks.repeat(1, self.n_emb).float()
        item_masks = item_masks.repeat(1, self.n_emb).float()

        user_masks = torch.where(user_masks == 0, torch.full_like(user_masks, 1e-10), user_masks)
        item_masks = torch.where(item_masks == 0, torch.full_like(item_masks, 1e-10), item_masks)

        user_avgs = torch.div(urecords_embs, user_masks)
        item_avgs = torch.div(irecords_embs, item_masks)
        user_avgs = self.emb_dropout(user_avgs)
        user_avgs = torch.reshape(user_avgs, (batch_size, self.T, self.n_emb))
        item_avgs = self.emb_dropout(item_avgs)
        item_avgs = torch.reshape(item_avgs, (batch_size, self.T, self.n_emb))

        user_avgs = torch.cat([user_avgs, uid_embs], dim=2)
        item_avgs = torch.cat([item_avgs, iid_embs], dim=2)

        uout, ustate = self.user_encoder(user_avgs)
        iout, istate = self.item_encoder(item_avgs)

        ustate = ustate.view(self.n_layer, 2, batch_size, self.gru_hidden)
        uforward_state, ubackward_state = ustate[-1][0], ustate[-1][1]
        u_final = torch.cat([uforward_state, ubackward_state], dim=1)
        istate = istate.view(self.n_layer, 2, batch_size, self.gru_hidden)
        iforward_state, ibackward_state = istate[-1][0], istate[-1][1]
        i_final = torch.cat([iforward_state, ibackward_state], dim=1)
        y_final = torch.cat([u_final, i_final], dim=1)
        return self.out_fc(y_final)
