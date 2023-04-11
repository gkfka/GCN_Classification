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

def test(model, dataset):
    model.eval()
    preds = []

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

            max_preds = logits.argmax(dim = -1)
            preds += max_preds.tolist()
            # correct = (tgt_label == max_preds).int().sum()

    return preds

if __name__ == "__main__":
    max_depend_len = 200
    data_fnm = "./data/test.csv"
    # model_fnm = "./model_save/nomal_best_model.pt.pt"
    model_fnm = "./checkpoint.pt"
    dp_fnm = "./data/test.txt"
    BATCH_SIZE = 128

    tag_data = dict(read_json("tag.json"))

    label_data = pd.read_csv(data_fnm, encoding='utf-8')
    for key, value in tag_data.items():
        if len(label_data[label_data['tag'].isin([key])]) == 0: continue
        label_data['tag'] = label_data['tag'].replace([key],[value])

    sentences = label_data['sentenct'].values
    tags =  label_data['tag'].values
    test_data = Classi_Dataset(sentences, tags, dp_fnm)

    collator = Classi_Collator(max_depend_len)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PaperClassifier(output_dim=len(tag_data), max_depend_len=max_depend_len)
    # model.load_state_dict(torch.load(model_fnm))
    checkpoint = torch.load(model_fnm)
    print(checkpoint.keys())
    new_state_dict = OrderedDict()

    # for k, v in checkpoint['model_state_dict'].items():
    for k, v in checkpoint.items():
        name = k[7:] # remove `module.`  ## module 키 제거
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    
    print("\nTesting...")
    start_time = timer()
    preds = test(model, test_data)
    end_time = timer()

    print(f"test time = {(end_time - start_time):.3f}s")

    print("예측\t\t정답\t\t문장")
    reverse_tag_dict = { tag_data[k]:k for k in tag_data }
    for s, t, p in zip(sentences, tags, preds):
        print(f"{reverse_tag_dict[p]}\t{reverse_tag_dict[t]}\t{s}")
