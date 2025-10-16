# -*- coding: utf-8 -*-
from gensim.models import FastText

# Path dữ liệu
pathdata = './datatrain.txt'

# Hàm đọc dữ liệu
def read_data(path):
    traindata = []
    with open(path, 'r', encoding='utf-8') as f:
        for sent in f:
            traindata.append(sent.strip().split())
    return traindata

if __name__ == '__main__':
    train_data = read_data(pathdata)
    print(f"Đã đọc {len(train_data)} câu huấn luyện.")

    # Tạo và huấn luyện mô hình FastText
    model = FastText(
        vector_size=150,  
        window=10,       
        min_count=2,      
        workers=4,       
        sg=1              
    )

    model.build_vocab(corpus_iterable=train_data)
    model.train(
        corpus_iterable=train_data,
        total_examples=len(train_data),
        epochs=5
    )

    # Lưu mô hình
    model.wv.save("fasttext_gensim.model")

    # Ví dụ từ gần nghĩa
    if "công_nghệ" in model.wv:
        print("5 từ gần nghĩa với 'công_nghệ':")
        for word, score in model.wv.most_similar("công_nghệ", topn=5):
            print(word, round(score, 4))
