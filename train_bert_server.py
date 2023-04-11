import csv
import pandas as pd
import numpy as np
from utils import *
from dataset import *
from torch.utils.data import DataLoader
# from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from timeit import default_timer as timer
from pytorchtools import EarlyStopping
from model import *

import korscibert_pytorch.modeling as modeling

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "-1"


PAD_IDX = 0

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

def create_mask(src):
    src_seq_len = src.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    return src_mask, src_padding_mask


def attention_mask(ids):
    masks = []
    for id in ids:
        mask = [float(i>0) for i in id]
        masks.append(mask)
    return torch.tensor(masks)

def save( epoch, model, optimizer, losses, train_step,model_name):
    torch.save({
        'epoch': epoch,  # 현재 학습 epoch
        'model_state_dict': model.state_dict(),  # 모델 저장
        'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
        'losses': losses,  # Loss 저장
        'train_step': train_step,  # 현재 진행한 학습
    }, f'{model_name}.pt')

def train_epoch(model, optimizer, dataset, scheduler):
    model.train()
    losses = 0
    print_loss = 0
    accuracy = 0
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collator.collate, shuffle=True)
    c_matrix_p = torch.zeros((9,9),dtype=torch.float)
    c_matrix_r = torch.zeros((9,9),dtype=torch.float)
    
    for i, data in enumerate(dataloader):
        src_seg = data["src_seg"].contiguous().to(DEVICE)
        tgt_tag = data["tgt_tag"].contiguous().to(DEVICE)
        src_adj = data["src_adj"].contiguous().to(DEVICE)
        src_node_mask = data["src_node_mask"].contiguous().to(DEVICE)

        src_input_ids = data["src_input_ids"].contiguous().to(DEVICE)
        src_input = src_input_ids.transpose(0, 1).contiguous().to(DEVICE)

        # _, src_padding_mask = create_mask(src_input)
        src_mask = attention_mask(src_input).transpose(0,1).contiguous().to(DEVICE)
        tgt_label = data["tgt_tag"].to(DEVICE)

        output, loss = model(src_input_ids, src_mask, src_seg, tgt_label, src_adj, src_node_mask)
        loss = loss.sum()

        # loss = output.loss

        losses += loss.item()
        print_loss += (loss.item() - print_loss) / (i + 1)

        loss.backward()

        max_preds = output.argmax(dim = -1)
        correct = (tgt_label == max_preds).int().sum()  #CROSSENTROPYLOSS
        # accuracy += correct.sum()/len(data["src_input_ids"])/len(dataloader)
        
        accuracy += (correct.sum()/len(data["src_input_ids"])-accuracy) / (i + 1)
        count_c_matrix(tgt_label,max_preds,c_matrix_p,c_matrix_r)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        # if (i+1) % 10 == 0:
        #     print("\nStep :", i+1)
        #     print(max_preds)
        #     print(tgt_label)
        #     prec,re,f1_score,macro_f1 = compute_performance(c_matrix_p,c_matrix_r)
        #     print_performace(print_loss, correct.sum()/len(data["src_input_ids"]), prec,re,f1_score,macro_f1,c_matrix_p)
    
    return losses / len(dataloader)
# 정확도 계산 함수
# def flat_accuracy(preds, labels):
    
#     pred_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()

#     return np.sum(pred_flat == labels_flat) / len(labels_flat)


def evaluate(model, dataset):
    model.eval()
    losses = 0
    class_num = 9
    correct_cnt = 0
    data_cnt = 0
    accuracy = 0
    c_matrix_p = torch.zeros((class_num,class_num),dtype=torch.float)
    c_matrix_r = torch.zeros((class_num,class_num),dtype=torch.float)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collator.collate)
    with torch.no_grad():
        for data in dataloader:
            src_seg = data["src_seg"].to(DEVICE).contiguous()
            tgt_tag = data["tgt_tag"].to(DEVICE).contiguous()
            src_adj = data["src_adj"].to(DEVICE).contiguous()
            src_node_mask = data["src_node_mask"].to(DEVICE).contiguous()

            src_input_ids = data["src_input_ids"].to(DEVICE).contiguous()
            src_input = src_input_ids.transpose(0, 1)

            # _, src_padding_mask = create_mask(src_input)
            src_mask = attention_mask(src_input).transpose(0,1).contiguous().to(DEVICE)
            tgt_label = data["tgt_tag"].to(DEVICE).contiguous()

            logits, loss = model(src_input_ids, src_mask, src_seg, tgt_label, src_adj, src_node_mask)
            loss = loss.sum()
            losses += loss.item()
            # print(logits)

            # CPU로 데이터 이동
            # logits = logits.detach().cpu().numpy()
            # tgt_label = tgt_label.to('cpu').numpy()

            # 출력 로짓과 라벨을 비교하여 정확도 계산
            # logits
            max_preds = logits.argmax(dim = -1)
            correct = (tgt_label == max_preds).int().sum()  #CROSSENTROPYLOSS
            correct_cnt += correct.sum()
            data_cnt += len(data["src_input_ids"])
            # accuracy += correct.sum()/len(data["src_input_ids"])/len(dataloader)
            count_c_matrix(tgt_label, max_preds, c_matrix_p, c_matrix_r)
    accuracy = correct_cnt / data_cnt
    prec,re,f1_score,macro_f1 = compute_performance(c_matrix_p,c_matrix_r)
    
    return losses / len(dataloader), accuracy,prec,re,f1_score,macro_f1,c_matrix_p

def print_performace(losses, accuracy,prec,re,f1_score,macro_f1,c_matrix_p):
    """
    losses, accuracy, macro_f1 == 값
    prec, re, f1_score == 1차원 벡터
    """
    class_num = 9
    accuracy = accuracy * 100
    macro_f1 = macro_f1 * 100
    # print("-----confusion_matrix-----")
    # print('--------------------------answer--------------------------')
    # for i in range(class_num):
    #     print(f'| {c_matrix_p[i]}|')
    print("-side-is-pred---------------------------------------------")
    print()
    print(f"precision : {prec}")
    print(f"recall    : {re}")
    print(f"macro_f1  : {f1_score}")
    print()
    print("--------------performance--------------")
    print(f"|losses   : {losses:.3f}  |")
    print(f"|accuracy : {accuracy:.1f}   |")
    print(f"|macro_f1 : {macro_f1:.1f}   |")
    print("--------------------------------------")

if __name__ == "__main__":
    max_depend_len = 200
    data_fnm = "./data/tagging.csv"
    dp_fnm = "./data/tagging_rslt.txt"

    tag_data = dict(read_json("tag.json"))
    print(tag_data)

    label_data = pd.read_csv(data_fnm, encoding='utf-8')
    label_data = label_data[28000:]
    for key, value in tag_data.items():
        label_data['tag'] = label_data['tag'].replace([key],[value])

    train_sentences, val_sentences = [], []
    train_tags, val_tags = [], []
    for i in range(len(tag_data)):
        
        aa = label_data.loc[(label_data['tag'] == i)]
        class_len = len(aa)

        split_num = int(class_len * 0.9)
        tmp_sen = aa['sentenct'].values
        tmp_tag = aa['tag'].values

        for idx, sen in enumerate(tmp_sen):
            if idx < split_num:
                train_sentences.append(sen)
                train_tags.append(tmp_tag[idx])
            else:
                val_sentences.append(sen)
                val_tags.append(tmp_tag[idx])


        

    # sentences = label_data['sentenct'].values
    # tags =  label_data['tag'].values
    # train_data = Classi_Dataset(sentences[:10000],tags[:10000])
    # val_data = Classi_Dataset(sentences[10000:],tags[10000:])
    train_data = Classi_Dataset(train_sentences,train_tags,dp_fnm)
    val_data = Classi_Dataset(val_sentences,val_tags,dp_fnm)


    collator = Classi_Collator(max_depend_len)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cpu')

    # dataloader = DataLoader(train_data, batch_size=16, collate_fn=collator.collate)
    # for data in dataloader:
    #     continue
    # dataloader = DataLoader(val_data, batch_size=16, collate_fn=collator.collate)
    # for data in dataloader:
    #     continue
    # exit(1)

    BATCH_SIZE = 64
    NUM_EPOCHS = 100

    start_epoch = 1
    global_steps = 0
    train_dataset_length = len(train_data)
    total_steps = len(train_data) * NUM_EPOCHS

    bert_config = modeling.BertConfig.from_json_file("./korscibert_pytorch/bert_config_kisti.json")

    # model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=9)
    model = PaperClassifier(output_dim=9, max_depend_len=max_depend_len)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    # scheduler = None

    MODEL_PATH = './model_save/nomal_best_model.pt'
    patience = 5
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    best_valid_loss = -1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        start_time = timer()

        print("Epoch", epoch)
        print("Training...")
        train_loss = train_epoch(model, optimizer, train_data, scheduler)
        print("Evaluating...")
        valid_loss, accuracy, prec, re, f1_score, macro_f1, c_matrix_p = evaluate(model, val_data)
        end_time = timer()

        
        print((
                    f"Epoch: {epoch},Train loss: {train_loss:.6f}, Valid loss: {valid_loss:.6f}, Epoch time = {(end_time - start_time):.3f}s"))
        print_performace(valid_loss, accuracy, prec, re, f1_score, macro_f1, c_matrix_p)
        # print((
        #             f"Epoch: {epoch},Train loss: {train_loss:.6f}, Valid loss: {valid_loss:.6f}, Epoch time = {(end_time - start_time):.3f}s"))
        # print((
        #             f"Valid acc: {f1_score:.6f}, Valid macro_f1: {macro_f1:.6f}"))
            
        global_steps += 1

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping!!")
            break

        if best_valid_loss == -1: best_valid_loss=valid_loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"{epoch}에 저장했당!")

            save(epoch,model,optimizer,train_loss,global_steps ,MODEL_PATH)
