
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel
from gnn import *
from utils import *

class PaperClassifier(nn.Module):
    def __init__(self, output_dim, max_depend_len): 
        super().__init__()
        self.n_labels   = output_dim
        self.rnn        = nn.GRU(768, 768, batch_first=True, bidirectional=True)
        self.fc_out     = nn.Linear(768*2, output_dim)
        # self.fc_out     = nn.Linear(768, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.max_depend_len = max_depend_len

        self.gc1 = GraphConvolution(768, 768)#self.max_depend_len)
        self.gc2 = GraphConvolution(768, 768)#self.max_depend_len)
        # self.gc3 = GraphConvolution(768, 768)#self.max_depend_len)

        self.config = BertConfig.from_pretrained('./korscibert_pytorch/bert_config_kisti.json')
        self.BERTmodel = BertModel.from_pretrained('./model/pytorch_model.bin', config=self.config)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        hidden_output = hidden_output.double()
        e_mask = e_mask.double()
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        length_tensor = length_tensor+(length_tensor == 0).float()

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.double(), hidden_output).squeeze(1)
        avg_vector = sum_vector.double() / length_tensor.double()  # broadcasting
        return avg_vector.float()

    def forward(self, src_input_ids, src_mask, src_seg, tgt_tag, src_adj, src_node_mask):
        # src_mask = src_mask.transpose(0,1).contiguous()
        # print(src_input_ids.shape, src_mask.shape, src_seg.shape)
        emb = self.BERTmodel(src_input_ids, attention_mask=src_mask, token_type_ids=src_seg)
        
        # print(src_input_ids.shape, emb.last_hidden_state.shape)
        # print(src_mask.shape, src_seg.shape)
        # print(tgt_tag.shape, src_adj.shape)
        # print(src_node_mask.shape)
        ###############
        depend_features = self.get_depend_features(emb.last_hidden_state, src_node_mask)

        # print(depend_features.shape, src_adj.shape)
        # input()
        gcn_output = []
        for df, am in zip(depend_features, src_adj):
            gcn_out = self.gc1(df, am)
            gcn_out = self.gc2(gcn_out, am)
            # gcn_out = self.gc3(gcn_out, am)
            gcn_output.append(gcn_out)
        gcn_output = torch.stack(gcn_output)
        # print(gcn_output.shape)

        _, hidden = self.rnn(
            gcn_output,
            torch.stack([emb.pooler_output, emb.pooler_output])
        )
        # print(rout.shape, hidden.shape)
        # input()
        hidden = hidden.transpose(0, 1).contiguous()
        hidden = self.dropout(hidden)
        hidden = hidden.view(len(src_input_ids), -1)
        #################

        # print(hidden.shape)

        y_out = self.fc_out(hidden)
        
        # y_out = self.fc_out(emb.pooler_output)

        # y_out = F.softmax(y_out, dim=1)
        # loss = F.cross_entropy(y_out, tgt_tag)
        
        loss_type = "focal"
        beta = 0.9999
        gamma = 2.0
        loss_fct = CB_loss(beta=beta, gamma=gamma)
        loss = loss_fct(
            y_out.view(-1, self.n_labels), tgt_tag, loss_type#tgt_tag.view(-1), loss_type
        )

        return y_out, loss
        
    def get_depend_features(self, sequence_output, depend_mask):
        depend_features = []

        for i in range(1, self.max_depend_len+1):
            avg_mask = (depend_mask == i).float()

            dep_avg_h = torch.nan_to_num(self.entity_average(sequence_output, avg_mask))

            depend_features.append(dep_avg_h)

        return torch.stack(depend_features, dim=1)
