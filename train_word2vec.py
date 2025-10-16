import os
from gensim.models import Word2Vec


PATH_DATA = './datatrain.txt'
PATH_MODEL = './model/word2vec_skipgram.model'


def read_data(path):
    
    traindata = []
    with open(path, 'r', encoding='utf-8') as f:
        for sent in f:
            sent = sent.strip()
            if sent:  
                traindata.append(sent.split())
    return traindata


if __name__ == '__main__':
    print("Đang đọc dữ liệu từ:", PATH_DATA)
    train_data = read_data(PATH_DATA)
    print(f"Đã đọc {len(train_data)} câu huấn luyện.")

  
    print("Đang huấn luyện Word2Vec...")
    model = Word2Vec(
        sentences=train_data,
        vector_size=150,  
        window=10,        
        min_count=2,      
        workers=4,        
        sg=1              
    )

   
    os.makedirs(os.path.dirname(PATH_MODEL), exist_ok=True)

    
    model.wv.save(PATH_MODEL)
    print(f"Huấn luyện xong! Mô hình được lưu tại: {PATH_MODEL}")

    
    try:
        print("\n5 từ gần nghĩa nhất với 'công_nghệ':")
        for w, score in model.wv.most_similar("công_nghệ", topn=5):
            print(f"{w}: {score:.4f}")
    except KeyError:
        print("Từ 'công_nghệ' chưa có trong từ điển của mô hình.")