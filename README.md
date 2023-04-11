# GCN_Classification

Training Enviroment

- CUDA 10.2
- NVIDIA GeForce GTX 1080 Ti

1. Pre-processing

1-1 json에서 필요한 데이터만 추출

*tag에 “분석방법”이 없고 “데이터처리”가 존재하여 이를 치환

- json파일에 존재하는 "\r\n"을 제거
- 문장과 태그만 추출하여 csv파일로 저장
- 의존구문 분석기를 위한 txt파일 저장

- test를 위한 새로운 data json파일로 이용시 extract.py안의 파일명 수정 필요
- CSV_FILE = "parser/data/tagging.csv" → "parser/data/tagging_test.csv"
TXT_FILE = "parser/data/tagging.txt" → "parser/data/tagging_test.txt"

```python
python extract.py
```

1-2 의존구문 분석

- 의존구문 분석을 위한 가상 환경 필요
- 의존구문 분석 시간 다소 필요
- (서버에 가상환경 만들어 져 있기 때문에 activate만 하면 됨)

```python
conda create -n Dpar python=3.6
conda activate Dpar
conda install pytorch=0.4.1 cuda100 -c pytorch
conda install numpy=1.15.4
pip install gensim
conda install dataclasses

#cd "의존구문 분석기가 존재하는 폴더"
cd parser
python bin/main.py -root_dir ./ -file_name "의존구문을 시행할 txt" -batch_size 30 -save_file "의존 구문 분석 결과 txt" -use_gpu
#train을 위한 의존구문 분석 명령어
#python bin/main.py -root_dir ./ -file_name "./data/tagging.txt" -batch_size 30 -save_file "./result/tagging_rslt.txt" -use_gpu
#test를 위한 의존구문 분석 명령어
#python bin/main.py -root_dir ./ -file_name "./data/tagging_test.txt" -batch_size 30 -save_file "./result/tagging_test_rslt.txt" -use_gpu

```

- 의존구문 분석이 끝나면 다시 conda activate classi 해줘야함
- json파일에서 ??????로 존재하는 문장은 의존구문 분석에서 ERROR가 나기 때문에 인접 행렬 구성 시 초기화 값을 이용

1-3 모델 가상 환경 필요

```python
conda create -n classi python=3.6
conda activate classi
pip install torch==1.8.1
pip install transformers==4.7.0
pip install tensorflow==1.15.0

pip install pandas
pip install konlpy
# Mecab설치
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar xvfz mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make check
make install
sudo ldconfig
mecab --version
#mecab-ko-dic설치
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
./configure
make
make install
sudo apt install curl
sudo apt install git
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
pip install mecab-python

```

1. Model

2-1. 문서 분류(Classification)모델

- [extract.py](http://extract.py) : json파일에서 문장과 태그를 읽어 csv, txt로 저장
- [gnn.py](http://gnn.py) : GCN구현
- [model.py](http://model.py) : classification 모델이 구현
- [dataset.py](http://dataset.py) : 논문 문장과 class를 담은 dataset
- [pytorchtools.py](http://pytorchtools.py) : early stopping 구현
- tag.json : tag와 숫자를 매핑
- [utils.py](http://utils.py) : 필요한 함수 구현

```
├── data
│   └──tagging_train_result.json
│
├── korscibert_pytorch
│   
├── model_save
│   └── best_model.pt
├── parser
│   ├── bin
│		│   └── main.py
│   ├── data
│		│   ├── tagging.txt // 배포된 전체 data
│		│   ├── tagging.csv // 배포된 전체 data
│		│   ├── tagging_test.txt //임의의 test 28000
│		│   ├── tagging_test.csv //임의의 test 28000
│		│   ├── tagging_.txt //임의의 test 28000을 제외한 train/val data
│		│   └── tagging_.csv //임의의 test 28000을 제외한 train/val data
│		│
│   ├── result
		│   ├── tagging_test_rslt.txt //임의의 test 28000 의존구문분석 결과
│		│   └── tagging_rslt.txt //임의의 test 28000을 제외한 train/val data 의존구문분석
│   ├── models
│   ├── datas.py
│   └── utils.py
│
├── dataset.py
├── extract.py
├── gnn.py
├── inference.py
├── model.py
├── pytorchtools.py
├── tag.json
├── train_bert_server.py
└── utils.py

```

- korsciBERT 이용하여 제공 받은 데이터를 임베딩
- 의존구문 분석 그래프를 이용하여 인접행렬 구성
- 구성한 인접행렬과 GCN(Graph Convolutional Network)을 이용하여 class예측
- 각 class의 불균형의 존재 —> focal loss이용, macro/micro fi-score 이용

1. How to use 🔅 
    
    **Trainig**
    
    - 의존구문 분석기 실행
    - train 파일 실행
    
    ```python
    conda activate Dpar
    cd parser
    python bin/main.py -root_dir ./ -file_name "./data/tagging_.txt" -batch_size 30 -save_file "./result/tagging_rslt.txt" -use_gpu
    cd ..
    conda activate classi
    
    python train_bert_server.py
    
    ```
    
    - model_save에 best model이 저장
    
    ```python
    #입력"sentence, tag" csv 데이터
    data_fnm = "./parser/data/tagging.csv"
    #의존구문 분석된 데이터
    dp_fnm = "./parser/result/tagging_rslt.txt"
    ```
    
    **Evaluate**
    
    - model_save의 모델을 읽어온 후 의존구문 분석이 끝난 test파일을 test
    - 성능이 출력
    
    ```python
    #입력"sentence, tag" csv 데이터
    data_fnm = "./parser/data/tagging_test.csv"
    #저장된 best 모델
    model_fnm = './model_save/best_model.pt'
    #의존구문 분석된 데이터
    dp_fnm = "./parser/result/tagging_test_rslt.txt"
    ```
    
    ```python
    conda activate classi
    python inference.py
    ```
    
    **Demo**
    
    test에서 랜덤 추출한 문장 : corpus.csv
    
    ```python
    conda activate Dpar
    cd parser
    python bin/main.py -root_dir ./ -file_name "./data/corpus.txt" -batch_size 30 -save_file "./result/corpus_rslt.txt" -use_gpu
    cd ..
    conda activate classi
    python demo.py
    
    ```
    
    - 문장의 각 클래스가 예측되어 나옴
    
