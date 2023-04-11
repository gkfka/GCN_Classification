import json
import pickle
import copy
import random
from pprint import pprint

def read_corpus(fnm, get_guid=False):
    sent = []

    with open(fnm, "r", encoding="utf-8") as f:
        data_lst = json.load(f)

    for data in data_lst:
        guid = data["guid"]
        text = data["sentence"]
        subject_entity = data["subject_entity"]
        object_entity = data["object_entity"]
        label = data["label"]

        if get_guid:
            sent.append((guid, text))
        else:
            sent.append(text)

    print("File Read Complete !")
    print(len(sent), "sents Loaded")

    return sent

def read_conll(fnm):
    sents = []

    with open(fnm, "r", encoding="utf-8") as f:
        data_lst = f.read().split("\n\n")

    for data in data_lst:
        word_list = data.split("\n")
        sent = word_list[0][7:]
        if len(sent) == 0: continue
        
        words = []
        for word_line in word_list[1:]:
            #print("!!!", word_line)
            word_split = word_line.split("\t")
            word, head, typ = word_split[1], word_split[-4], word_split[-3]
            words.append((word,head,typ))
        sents.append([sent, words])

    return sents

def write_pickle(pname, obj):
    with open(pname, "wb") as f:
        pickle.dump(obj, f)

def read_pickle(pname):
    with open(pname, "rb") as f:
        return pickle.load(f)

if __name__=='__main__':
    ###
    # sent = read_corpus("./data/klue-re-v1.1_dev.json")
    
    # with open("./data/dev.txt", "w+", encoding="utf-8") as f:
    #    for s in sent:
    #        print(s, file=f)

    ###
    # sents = read_conll("./data/result_example.txt")
    # print(sents)

    
    depend_words = read_pickle("./sents_dict_train.pickle")
    max_len = 0
    for guid in depend_words:
        if len(depend_words[guid]["depend"]) > max_len:
            max_len = len(depend_words[guid]["depend"])
    print(max_len)
    
    print("\n----------------------\n")
    depend_words = read_pickle("./sents_dict_dev.pickle")
    max_len = 0
    for guid in depend_words:
        if len(depend_words[guid]["depend"]) > max_len:
            max_len = len(depend_words[guid]["depend"])
    print(max_len)




    
    
