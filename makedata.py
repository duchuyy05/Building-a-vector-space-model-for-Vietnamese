from pyvi import ViTokenizer
import re, string
import pandas as pd

# === Hàm xử lý văn bản ===
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)        # bỏ HTML tag
    text = re.sub(r'\s+', ' ', text).strip() # chuẩn hóa khoảng trắng
    return text

def normalize_text(text):
    # loại bỏ dấu câu (trừ '_')
    listpunct = string.punctuation.replace('_', '')
    for p in listpunct:
        text = text.replace(p, ' ')
    return text.lower().strip()

# === Load stopword ===
stopwords = pd.read_csv('./stopwords.csv', sep='\t', encoding='utf-8')['stopwords'].tolist()

def remove_stopword(text):
    return ' '.join([w for w in text.split() if w not in stopwords])

# === Đọc dữ liệu gốc ===
data = pd.read_csv(r'C:\venv\prj\NLP_W5\corpus-title.csv', header=None, on_bad_lines='skip')
texts = data.iloc[:, 0].dropna().astype(str).tolist()

# === Ghi ra file train ===
with open('datatrain.txt', 'w', encoding='utf-8') as f:
    for sent in texts:
        sent = clean_text(sent)
        sent = normalize_text(sent)
        sent = ViTokenizer.tokenize(sent)
        sent = remove_stopword(sent)
        if len(sent.split()) > 1:
            f.write(sent + '\n')

print("✅ Done! File datatrain.txt đã được tạo.")
