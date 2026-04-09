df_sample = df.sample(n=50000, random_state=42)
texts = df_sample['combined_text']
y = df_sample['label'].values

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9
)

X = tfidf.fit_transform(texts)

print(X.shape)

X_tfidf = tfidf_vectorizer.fit_transform(texts)
y = df['label'].values
