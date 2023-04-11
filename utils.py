import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def read_json(fname):
    with open(fname, 'rt', encoding="UTF8") as f:
        return json.load(f)

def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(
            -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
        )

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

class CB_loss(nn.Module):
    def __init__(self, beta, gamma, epsilon=0.1):
        super(CB_loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits, labels, loss_type="softmax"):
        # self.epsilon = 0.1 #labelsmooth
        beta = self.beta
        gamma = self.gamma

        no_of_classes = logits.shape[1]
        samples_per_cls = torch.Tensor(
            [sum(labels == i) for i in range(logits.shape[1])]
        )
        if torch.cuda.is_available():
            samples_per_cls = samples_per_cls.cuda()

        effective_num = 1.0 - torch.pow(beta, samples_per_cls)
        weights = (1.0 - beta) / ((effective_num) + 1e-8)

        weights = weights / torch.sum(weights) * no_of_classes
        labels = labels.reshape(-1, 1)

        weights = torch.tensor(weights.clone().detach()).float()

        if torch.cuda.is_available():
            weights = weights.cuda()
            labels_one_hot = (
                torch.zeros(len(labels), no_of_classes)
                .cuda()
                .scatter_(1, labels, 1)
                .cuda()
            )

        labels_one_hot = (
            1 - self.epsilon
        ) * labels_one_hot + self.epsilon / no_of_classes
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, no_of_classes)

        if loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(
                input=logits, target=labels_one_hot, pos_weight=weights
            )
        elif loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(
                input=pred, target=labels_one_hot, weight=weights
            )
        return cb_loss

def count_c_matrix(ans,pred,c_p,c_r):
    answer_ = ans.clone().detach().int()
    pred_ = pred.clone().detach().int()
    for i in range(len(answer_)):
        c_p[pred_[i].item()][answer_[i].item()] += 1
        c_r[answer_[i].item()][pred_[i].item()] += 1
    # acc +=(answer_ == pred_).int().sum() / len(answer_)

def zero2num(in_num, zero_num=1):
    if in_num == 0: return zero_num
    return in_num

def compute_performance(c_p,c_r):
    class_num = 9
    prec = torch.zeros(class_num,dtype=torch.float)
    re = torch.zeros(class_num,dtype=torch.float)
    f1_score = torch.zeros(class_num,dtype=torch.float)

    for i in range(class_num):
        prec[i] = c_p[i][i] / zero2num(c_p[i].sum())
        re[i] = c_r[i][i] / zero2num(c_r[i].sum())
        f1_score[i] = 2*prec[i]*re[i]/zero2num(prec[i]+re[i])
    macro_f1 = f1_score.sum()/class_num

    return prec,re,f1_score,macro_f1
    
if __name__ == "__main__":
    print("hello world !")