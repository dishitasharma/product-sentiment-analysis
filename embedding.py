!pip install transformers torch
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

model.to(device)
model.eval()
texts = df['combined_text'].tolist()

def get_bert_embeddings(texts, batch_size=32, max_length=128):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

    df_sample = df.sample(50000, random_state=42)

texts = df_sample['combined_text'].tolist()
y = df_sample['label'].values

X = get_bert_embeddings(texts, batch_size=32)
np.save("/content/drive/MyDrive/bert_embeddings_50k.npy", X)
np.save("/content/drive/MyDrive/labels_50k.npy", y)
