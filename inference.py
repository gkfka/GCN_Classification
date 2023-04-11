import pandas as pd
from utils import *
from dataset import *
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from model import *
from collections import OrderedDict

def attention_mask(ids):
    masks = []
    for id in ids:
        mask = [float(i>0) for i in id]
        masks.append(mask)
    return torch.tensor(masks)

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
        for i, data in enumerate(dataloader):
            src_seg = data["src_seg"].contiguous().to(DEVICE)
            src_adj = data["src_adj"].contiguous().to(DEVICE)
            src_node_mask = data["src_node_mask"].contiguous().to(DEVICE)

            src_input_ids = data["src_input_ids"].contiguous().to(DEVICE)
            src_input = src_input_ids.transpose(0, 1).contiguous().to(DEVICE)

            # _, src_padding_mask = create_mask(src_input)
            src_mask = attention_mask(src_input).transpose(0,1).contiguous().to(DEVICE)
            tgt_label = data["tgt_tag"].to(DEVICE)

            logits, loss = model(src_input_ids, src_mask, src_seg, tgt_label, src_adj, src_node_mask)
            losses += loss.item()

            max_preds = logits.argmax(dim = -1)
            correct = (tgt_label == max_preds).int().sum()
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
    data_fnm = "./parser/data/tagging_test.csv"
    model_fnm = './model_save/best_model.pt'
    dp_fnm = "./parser/result/tagging_test_rslt.txt"
    BATCH_SIZE = 128

    tag_data = dict(read_json("tag.json"))

    label_data = pd.read_csv(data_fnm, encoding='utf-8')
    for key, value in tag_data.items():
        label_data['tag'] = label_data['tag'].replace([key],[value])

    sentences = label_data['sentenct'].values
    tags =  label_data['tag'].values
    test_data = Classi_Dataset(sentences, tags, dp_fnm)

    collator = Classi_Collator(max_depend_len)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PaperClassifier(output_dim=len(tag_data), max_depend_len=max_depend_len)

    checkpoint = torch.load(model_fnm)
    new_state_dict = OrderedDict()

    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:] # remove `module.`  ## module 키 제거
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    
    start_time = timer()

    print("Evaluating...")
    valid_loss, accuracy, prec, re, f1_score, macro_f1, c_matrix_p = evaluate(model, test_data)
    end_time = timer()

    print(f"test time = {(end_time - start_time):.3f}s")
    print_performace(valid_loss, accuracy, prec, re, f1_score, macro_f1, c_matrix_p)