from utils import *

if __name__=='__main__':
    ###
    # sent = read_corpus("./data/klue-re-v1.1_dev.json")
    
    # with open("./data/dev.txt", "w+", encoding="utf-8") as f:
    #    for s in sent:
    #        print(s, file=f)

    ###
    sents = read_corpus("./data/klue-re-v1.1_train.json", get_guid=True)
    sents = { s:guid for guid, s in sents}
    conll_sents = read_conll("./result/train_conll.txt")

    sents_dict = {}
    #cnt=0
    #for s in sents:
    #    cnt+=1
    #    print(s, sents[s])
    #    if cnt > 5:break
    for cs in conll_sents:
        #print(cs[0])
        sents_dict[sents[cs[0]]] = {"sent":cs[0],"depend":cs[1]}
        #print()
    #print(sents_dict)
    write_pickle("sents_dict_train.pickle", sents_dict)
    print(len(conll_sents), len(sents_dict))
