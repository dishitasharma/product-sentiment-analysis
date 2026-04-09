
import pandas as pd

def load_data(path):

    df = pd.read_json(path, lines=True)

    df = df[['rating', 'title', 'text', 'asin']]

    df = df.dropna(subset=['rating'])

    return df


def label_sentiment(rating):
    if rating >= 4:
        return 2  
    elif rating == 3:
        return 1  
    else:
        return 0   


def preprocess(df):

    df['title'] = df['title'].fillna("")
    df['text'] = df['text'].fillna("")

    
    df['combined_text'] = df['title'] + " " + df['text']

   
    df = df[df['combined_text'].str.strip() != ""]

   
    df['label'] = df['rating'].apply(label_sentiment)

  
    df = df.rename(columns={'asin': 'product_id'})

    return df

path = "/content/drive/MyDrive/All_Beauty.jsonl"

df = load_data(path)
df = preprocess(df)

print(df.head())
