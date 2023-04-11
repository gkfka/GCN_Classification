from torch.utils.data import Dataset
import korscibert_pytorch.tokenization_kisti as tokenization
import torch
import copy

def make_dp(sents, conll_sents):
    sents_dict = {}
    tmp = []
    sent = ""
    for s in conll_sents:
        if s[:5] == "#SENT":
            sent = s.strip()[7:]
        else:
            tok = s.strip().split('\t')
            if len(tok) != 1:
                tmp.append((tok[1],int(tok[-4])))
            else:
                # sents_dict["SENT"].append(sent)
                # sents_dict["DP"].append(tmp)
                sents_dict[sent] = tmp
                sent= ""
                tmp = []

    return [ sents_dict.get(s, None) for s in sents ]

class Classi_Dataset(Dataset):
    def __init__(self, sent, correct_tag, dp_fnm):
        self.sent = sent
        self.correct_tag = correct_tag    
        conll_sents = open(dp_fnm, encoding="utf-8").readlines() #나중에 파일 추가
        self.depend = make_dp(sent, conll_sents)

    def __len__(self):
        return len(self.sent)

    def __getitem__(self, idx):
        sent = self.sent[idx]
        correct_tag = self.correct_tag[idx]
        depend = self.depend[idx]
        return {"sent": sent, "correct_tag": correct_tag, "depend": depend }

class Classi_Collator:
    def __init__(self, dp_max_len):
        self.tokenizer = tokenization.FullTokenizer (vocab_file='./korscibert_pytorch/vocab_kisti.txt',do_lower_case=False,tokenizer_type="Mecab")

    def make_adj(self, MAX_LEN, sent, src_toks, depend):
        """
        MAX_LEN : 문장 최대 길이
        sent : 원문장
        src_tokens : 토크나이즈된 문장
        sents_dict : ("DP": [(어절, 의존구문정보),...], "SENT":[원문장,])
        pad_len : 패딩 길이
        """
        src_tokens = src_toks[:-1]

        adj, node_mask = [], [0]*len(src_tokens)
        for _ in range(MAX_LEN): adj+= [[0]*MAX_LEN]
        if MAX_LEN < len(sent.split(" ")):
            print(sent, MAX_LEN, len(sent.split(" ")))

        if depend == None:
            for i in range(len(sent.split(" "))):
                adj[i][i] = 1
                if i > 0:
                    adj[i-1][i] = 1
                    adj[i][i-1] = 1
        else:
            for i in range(len(depend)):
                adj[i][i] = 1
                _, head = depend[i]
                if head > 0:
                    adj[i][head-1] = 1
                    adj[head-1][i] = 1

        dep_idx = 0
        char_cnt = 0
        words = sent.split(" ")
        for i, tok in enumerate(src_tokens):
            if len(words[dep_idx]) <= char_cnt:
                dep_idx += 1
                char_cnt = 0
                if dep_idx >= len(words): break

            tok_len = len(tok)
            if tok[0]=="[" and tok[-1]=="]":
                tok_len = 1
            elif tok_len > 2 and tok[:2] == "##":
                tok_len -= 2

            char_cnt += tok_len

            node_mask[i] = dep_idx+1
        node_mask = [0] + node_mask + [0]

        return adj, node_mask

    def collate(self, data):
        src_adj, src_node_mask,src_texts, src_tokens, src_input_ids, src_mask, src_seg = [], [], [], [], [], [], []
        tgt_tag = []
        MAX_LEN = 512
        max_dep_len = 200
        for d in data:
            sent = d["sent"]
            correct_tag = d["correct_tag"]
            depend = d["depend"]

            src_texts.append(sent)
            tgt_tag.append(correct_tag)
            
            src_tokens = self.tokenizer.tokenize(sent)
            src_inut = self.tokenizer.convert_tokens_to_ids(src_tokens)
            if len(src_inut) > MAX_LEN-2:
                src_inut = src_inut[:MAX_LEN-2]
            add_special = copy.deepcopy(src_inut)
            add_special.insert(0, 6) #"[CLS]"
            add_special.append(7) #"[SEP]"

            segment_ids = [0] *len(add_special)
            input_mask = [1] *len(add_special)
            padding = [0]*(MAX_LEN-len(add_special))

            add_special += padding
            input_mask += padding
            segment_ids += padding


            # src_inut = self.tokenizer.convert_tokens_to_ids(src_tokens)
            src_input_ids.append(torch.LongTensor(add_special))
            src_tokens.append(src_tokens)
            src_mask.append(torch.LongTensor(input_mask))
            src_seg.append(torch.LongTensor(segment_ids))

            adj_matrix, node_mask = self.make_adj(max_dep_len, sent, src_tokens, depend)
            node_mask += padding
            src_adj.append(torch.FloatTensor(adj_matrix))
            src_node_mask.append(torch.LongTensor(node_mask))
        
        return {
            "src_text": src_texts,
            "src_tokens": src_tokens,
            "src_mask" : torch.stack(src_mask).contiguous(),
            "src_seg": torch.stack(src_seg).contiguous(),
            "src_input_ids": torch.stack(src_input_ids).contiguous(),
            "tgt_tag" : torch.tensor(tgt_tag).contiguous(),
            "src_adj" : torch.stack(src_adj).contiguous(),
            "src_node_mask" : torch.stack(src_node_mask).contiguous(),
        }
