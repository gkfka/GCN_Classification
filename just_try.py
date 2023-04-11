import korscibert_pytorch.tokenization_kisti as tokenization
vocab_file = "./vocab_kisti.txt"


tokenizer = tokenization.FullTokenizer (

vocab_file=vocab_file,

do_lower_case=False,

tokenizer_type="Mecab"

)


example = "본 고안은 주로 일회용 합성세제액을 집어넣어 밀봉하는 세제액포의 내부를 원호상으로 열중착하되 세제액이 배출되는 절단부 쪽으로 내벽을 협소하게 형성하여서 내부에 들어있는 세제액을 잘짜질 수 있도록 하는 합성세제 액포에 관한 것이다."

tokens = tokenizer.tokenize(example)

encoded_tokens = tokenizer.convert_tokens_to_ids(tokens)

decoded_tokens = tokenizer.convert_ids_to_tokens(encoded_tokens)


print("Input example ===>", example)

print("Tokenized example ===>", tokens)

print("Converted example to IDs ===>", encoded_tokens)

print("Converted IDs to example ===>", decoded_tokens)