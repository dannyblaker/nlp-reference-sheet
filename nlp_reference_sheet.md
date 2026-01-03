# Natural Language Processing (NLP) reference Sheet
## Author: Danny Blaker
---

## PAGE 1: Core Concepts & Text Processing

### **NLP Task Categories**
| Task | Description | Examples |
|------|-------------|----------|
| **Classification** | Assign labels to text | Sentiment analysis, spam detection, topic classification |
| **Sequence Labeling** | Label each token | NER, POS tagging, chunking |
| **Generation** | Produce new text | Translation, summarization, dialogue |
| **Extraction** | Pull out information | Entity extraction, keyword extraction, relation extraction |
| **Similarity** | Compare texts | Semantic search, duplicate detection, paraphrase detection |
| **Question Answering** | Find answers | Extractive QA, generative QA, open-domain QA |

### **Text Preprocessing Pipeline**
```python
# 1. LOWERCASING
text = text.lower()

# 2. TOKENIZATION
tokens = text.split()  # Basic
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)  # Better

# 3. REMOVE PUNCTUATION & SPECIAL CHARS
import re, string
text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

# 4. REMOVE STOPWORDS
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if w not in stop_words]

# 5. STEMMING vs LEMMATIZATION
from nltk.stem import PorterStemmer, WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stems = [stemmer.stem(w) for w in tokens]  # running → run
lemmas = [lemmatizer.lemmatize(w, pos='v') for w in tokens]  # running → run

# 6. NORMALIZATION
text = text.strip().replace('\n', ' ')  # Remove extra spaces
```

### **Feature Extraction Methods**
| Method | Description | Pros | Cons | Code |
|--------|-------------|------|------|------|
| **Bag of Words (BoW)** | Word frequency counts | Simple, interpretable | No word order, sparse | `CountVectorizer()` |
| **TF-IDF** | Term frequency × inverse doc freq | Weights important words | Still sparse, no semantics | `TfidfVectorizer()` |
| **N-grams** | Sequences of n words | Captures local context | Exponential feature growth | `ngram_range=(1,3)` |
| **Word2Vec** | Dense word embeddings | Semantic similarity | Needs large corpus | `gensim.Word2Vec()` |
| **GloVe** | Global vectors | Pre-trained available | Fixed vocabulary | `load_glove_embeddings()` |
| **FastText** | Subword embeddings | Handles OOV words | Larger model size | `fasttext.train()` |

```python
# BOW & TF-IDF with sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
bow = CountVectorizer(max_features=5000, ngram_range=(1,2))
X_bow = bow.fit_transform(corpus)
tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
X_tfidf = tfidf.fit_transform(corpus)

# Word2Vec
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
vector = model.wv['word']
similar = model.wv.most_similar('word', topn=10)
```

### **Key NLP Libraries**
| Library | Purpose | Installation |
|---------|---------|--------------|
| **NLTK** | Classic NLP toolkit | `pip install nltk` |
| **spaCy** | Production NLP | `pip install spacy && python -m spacy download en_core_web_sm` |
| **Transformers** | State-of-the-art models | `pip install transformers` |
| **Gensim** | Topic modeling, embeddings | `pip install gensim` |
| **TextBlob** | Simple NLP tasks | `pip install textblob` |
| **Stanford CoreNLP** | Java-based NLP suite | Download from Stanford |
| **Stanza** | Multilingual NLP | `pip install stanza` |

### **spaCy Quick Reference**
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Tokenization & attributes
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, 
          token.shape_, token.is_alpha, token.is_stop)

# Named Entity Recognition
for ent in doc.ents:
    print(ent.text, ent.label_, spacy.explain(ent.label_))

# Noun chunks
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_)

# Similarity (needs larger model: en_core_web_md/lg)
doc1 = nlp("I like cats")
doc2 = nlp("I like dogs")
similarity = doc1.similarity(doc2)
```

### **Regular Expressions for NLP**
```python
import re
# Email extraction
emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
# URL extraction
urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
# Phone numbers
phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
# Hashtags
hashtags = re.findall(r'#\w+', text)
# Mentions
mentions = re.findall(r'@\w+', text)
```

---

## PAGE 2: Advanced Techniques & Models

### **Word Embeddings Comparison**
| Model | Type | Dim | Training | Context | OOV |
|-------|------|-----|----------|---------|-----|
| **Word2Vec CBOW** | Predictive | 100-300 | Predict word from context | Local window | No |
| **Word2Vec Skip-gram** | Predictive | 100-300 | Predict context from word | Local window | No |
| **GloVe** | Count-based | 50-300 | Matrix factorization | Global co-occurrence | No |
| **FastText** | Predictive | 100-300 | Skip-gram + subwords | Local window + char n-grams | Yes |
| **ELMo** | Contextual | 1024 | BiLSTM language model | Entire sentence | Yes |
| **BERT** | Contextual | 768/1024 | Transformer encoder | Bidirectional | Yes |

### **Transformer Architecture Components**
```
INPUT → EMBEDDING → POSITIONAL ENCODING → ENCODER/DECODER → OUTPUT

Key Components:
1. SELF-ATTENTION: Attention(Q,K,V) = softmax(QK^T/√d_k)V
2. MULTI-HEAD ATTENTION: Concat(head_1,...,head_h)W^O
3. POSITION-WISE FFN: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
4. LAYER NORM: LayerNorm(x + Sublayer(x))
5. POSITIONAL ENCODING: PE(pos,2i) = sin(pos/10000^(2i/d))
```

### **Popular Transformer Models**
| Model | Type | Parameters | Best For | Hugging Face ID |
|-------|------|------------|----------|-----------------|
| **BERT** | Encoder | 110M/340M | Classification, NER, QA | `bert-base-uncased` |
| **RoBERTa** | Encoder | 125M/355M | Better BERT training | `roberta-base` |
| **DistilBERT** | Encoder | 66M | Fast inference | `distilbert-base-uncased` |
| **ALBERT** | Encoder | 12M/235M | Parameter efficiency | `albert-base-v2` |
| **GPT-2** | Decoder | 117M-1.5B | Text generation | `gpt2` |
| **GPT-3/4** | Decoder | 175B+ | Few-shot learning | `gpt-3.5-turbo` (API) |
| **T5** | Encoder-Decoder | 60M-11B | Text-to-text tasks | `t5-base` |
| **BART** | Encoder-Decoder | 139M/406M | Summarization, translation | `facebook/bart-base` |
| **XLNet** | Encoder | 110M/340M | Permutation LM | `xlnet-base-cased` |
| **ELECTRA** | Encoder | 14M/110M | Efficient pre-training | `google/electra-small` |

### **Hugging Face Transformers Quick Start**
```python
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# PIPELINES (easiest)
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")[0]

ner = pipeline("ner", aggregation_strategy="simple")
entities = ner("Apple Inc. is in Cupertino")

qa = pipeline("question-answering")
answer = qa(question="What is NLP?", context="Natural Language Processing...")

summarizer = pipeline("summarization")
summary = summarizer("Long text...", max_length=50)

# MANUAL APPROACH (more control)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# FINE-TUNING TEMPLATE
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16,
    per_device_eval_batch_size=64, warmup_steps=500, weight_decay=0.01,
    logging_dir="./logs", evaluation_strategy="epoch"
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, 
                  eval_dataset=val_ds)
trainer.train()
```

### **Text Classification Algorithms**
| Algorithm | Type | Pros | Cons | sklearn Class |
|-----------|------|------|------|---------------|
| **Naive Bayes** | Probabilistic | Fast, works well with small data | Independence assumption | `MultinomialNB()` |
| **Logistic Regression** | Linear | Interpretable, fast | Linear decision boundary | `LogisticRegression()` |
| **SVM** | Kernel-based | Effective in high dims | Slow with large data | `SVC(kernel='linear')` |
| **Random Forest** | Ensemble | Handles non-linearity | Can overfit | `RandomForestClassifier()` |
| **XGBoost** | Gradient Boosting | High performance | Requires tuning | `XGBClassifier()` |
| **Neural Networks** | Deep Learning | Learn complex patterns | Needs lots of data | `MLPClassifier()` or PyTorch |

### **Sequence Models**
```python
# RNN (Vanilla)
import torch.nn as nn
rnn = nn.RNN(input_size=100, hidden_size=128, num_layers=2, batch_first=True)
output, hidden = rnn(input_tensor)

# LSTM (Long Short-Term Memory)
lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=2, 
               batch_first=True, dropout=0.2)
output, (hidden, cell) = lstm(input_tensor)

# GRU (Gated Recurrent Unit)
gru = nn.GRU(input_size=100, hidden_size=128, num_layers=2, batch_first=True)
output, hidden = gru(input_tensor)

# Bidirectional LSTM
bilstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=2, 
                 bidirectional=True, batch_first=True)
output, (hidden, cell) = bilstm(input_tensor)  # output: [batch, seq, 256]

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden, encoder_outputs):
        scores = torch.bmm(encoder_outputs, self.attn(hidden).unsqueeze(2))
        weights = torch.softmax(scores.squeeze(2), dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        return context, weights
```

### **Named Entity Recognition (NER)**
```python
# spaCy NER
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Elon Musk founded SpaceX in 2002")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# Custom NER training
import spacy
from spacy.training import Example
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")
ner.add_label("TECH_COMPANY")
# Training data format: (text, {"entities": [(start, end, label)]})
TRAIN_DATA = [("Apple releases new iPhone", {"entities": [(0, 5, "TECH_COMPANY")]})]
optimizer = nlp.begin_training()
for epoch in range(10):
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], sgd=optimizer)

# Transformers NER
from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER")
entities = ner("Elon Musk founded SpaceX")
```

### **Topic Modeling**
```python
# LDA (Latent Dirichlet Allocation)
from gensim import corpora
from gensim.models import LdaModel
dictionary = corpora.Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
lda = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
topics = lda.print_topics(num_words=5)

# NMF (Non-negative Matrix Factorization)
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(documents)
nmf = NMF(n_components=10, random_state=42)
W = nmf.fit_transform(X)
H = nmf.components_
```

---

## PAGE 3: Evaluation, Advanced Tasks & Best Practices

### **Evaluation Metrics**
| Metric | Formula | Use Case | Code |
|--------|---------|----------|------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Balanced classes | `accuracy_score(y_true, y_pred)` |
| **Precision** | TP/(TP+FP) | Minimize false positives | `precision_score()` |
| **Recall** | TP/(TP+FN) | Minimize false negatives | `recall_score()` |
| **F1 Score** | 2×(P×R)/(P+R) | Balance precision & recall | `f1_score()` |
| **F-beta** | (1+β²)×(P×R)/(β²P+R) | Weight recall β times more | `fbeta_score(beta=2)` |
| **ROC-AUC** | Area under ROC curve | Binary classification | `roc_auc_score()` |
| **Perplexity** | exp(-Σlog P(w)/N) | Language models | `model.perplexity()` |
| **BLEU** | Geometric mean of n-gram precision | Machine translation | `sentence_bleu()` |
| **ROUGE** | Recall-oriented n-gram overlap | Summarization | `rouge_score()` |
| **METEOR** | Harmonic mean with synonym matching | Translation | `meteor_score()` |

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

# For multi-class
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
```

### **Text Similarity Metrics**
```python
# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

# Jaccard Similarity
def jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

# Levenshtein Distance (edit distance)
from Levenshtein import distance
dist = distance("kitten", "sitting")

# Semantic Similarity with Transformers
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
emb1 = model.encode("This is a sentence")
emb2 = model.encode("This is another sentence")
similarity = util.cos_sim(emb1, emb2)
```

### **Sentiment Analysis Approaches**
```python
# 1. LEXICON-BASED (VADER)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores("This product is amazing!")
# {'neg': 0.0, 'neu': 0.323, 'pos': 0.677, 'compound': 0.6588}

# 2. TEXTBLOB
from textblob import TextBlob
blob = TextBlob("I love this!")
sentiment = blob.sentiment  # Sentiment(polarity=0.5, subjectivity=0.6)

# 3. TRANSFORMERS
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("This movie was fantastic!")
```

### **Text Summarization**
```python
# EXTRACTIVE (select important sentences)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 3)  # 3 sentences

# ABSTRACTIVE (generate new text)
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(long_text, max_length=130, min_length=30, do_sample=False)
```

### **Question Answering**
```python
# Extractive QA
from transformers import pipeline
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
context = "The Eiffel Tower is in Paris, France. It was built in 1889."
question = "Where is the Eiffel Tower?"
answer = qa(question=question, context=context)
# {'answer': 'Paris, France', 'score': 0.98, 'start': 24, 'end': 37}

# Open-domain QA with retrieval
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
```

### **Machine Translation**
```python
# With Transformers
from transformers import pipeline
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
result = translator("Hello, how are you?")

# Manual with tokenizer
from transformers import MarianMTModel, MarianTokenizer
model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
text = "Hello world!"
inputs = tokenizer(text, return_tensors="pt", padding=True)
outputs = model.generate(**inputs)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### **Zero-Shot & Few-Shot Learning**
```python
# Zero-shot classification
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "This is a course about Python programming",
    candidate_labels=["education", "politics", "business"]
)

# Few-shot with GPT
import openai
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Classify sentiment as positive, negative, or neutral."},
        {"role": "user", "content": "Example: 'I love it!' -> positive"},
        {"role": "user", "content": "Example: 'I hate it!' -> negative"},
        {"role": "user", "content": "Classify: 'It was okay'"}
    ]
)
```

### **Advanced Techniques**
```python
# DATA AUGMENTATION
import nlpaug.augmenter.word as naw
aug = naw.SynonymAug(aug_src='wordnet')
augmented = aug.augment("The quick brown fox")

# ACTIVE LEARNING
from modAL.models import ActiveLearner
learner = ActiveLearner(estimator=classifier, X_training=X_train, y_training=y_train)
query_idx, query_inst = learner.query(X_pool)
learner.teach(X_pool[query_idx], y_pool[query_idx])

# TRANSFER LEARNING
from transformers import AutoModel
base_model = AutoModel.from_pretrained("bert-base-uncased")
# Freeze base layers
for param in base_model.parameters():
    param.requires_grad = False
# Add custom head
model = nn.Sequential(base_model, nn.Linear(768, num_classes))

# KNOWLEDGE DISTILLATION
from transformers import DistilBertForSequenceClassification
teacher = AutoModelForSequenceClassification.from_pretrained("bert-base")
student = DistilBertForSequenceClassification.from_pretrained("distilbert-base")
```

### **Production Best Practices**
| Practice | Description | Implementation |
|----------|-------------|----------------|
| **Model Versioning** | Track model versions | MLflow, DVC, W&B |
| **Batch Processing** | Process multiple texts | `DataLoader(batch_size=32)` |
| **Caching** | Cache embeddings/results | Redis, pickle, joblib |
| **A/B Testing** | Compare model versions | Feature flags, traffic splitting |
| **Monitoring** | Track performance drift | Prometheus, custom logging |
| **Error Handling** | Graceful degradation | Try-except, fallback models |
| **Optimization** | Speed up inference | ONNX, TensorRT, quantization |
| **API Design** | RESTful endpoints | FastAPI, Flask |

### **Performance Optimization**
```python
# MODEL QUANTIZATION
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base", torchscript=True)
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# ONNX EXPORT
import torch.onnx
dummy_input = torch.randn(1, 128)
torch.onnx.export(model, dummy_input, "model.onnx")

# MIXED PRECISION TRAINING
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
for data, target in dataloader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### **Common Pitfalls & Solutions**
| Issue | Cause | Solution |
|-------|-------|----------|
| **Overfitting** | Model memorizes training data | Dropout, regularization, more data |
| **Class Imbalance** | Unequal class distribution | SMOTE, class weights, under/oversampling |
| **OOV Words** | Unknown vocabulary | FastText, character embeddings, subword tokenization |
| **Long Sequences** | Memory constraints | Truncation, chunking, hierarchical models |
| **Domain Shift** | Train/test distribution mismatch | Domain adaptation, fine-tuning, data augmentation |
| **Low Resource** | Limited training data | Transfer learning, data augmentation, semi-supervised |

### **Quick Command Reference**
```bash
# Download models
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet

# Training
python train.py --model bert-base --epochs 10 --lr 2e-5 --batch-size 16

# Evaluation
python evaluate.py --model checkpoints/best_model --test-data test.csv

# Inference
python predict.py --input "Your text here" --model model.pkl

# Export model
transformers-cli convert --model_type bert --pytorch_model_path model.bin \
  --config config.json --tf_dump_path tf_model
```

### **Extra Resources**
- **Papers**: [arXiv.org](https://arxiv.org), [Papers with Code](https://paperswithcode.com)
- **Datasets**: HuggingFace Datasets, Kaggle, UCI ML Repository, Common Crawl
- **Models**: [HuggingFace Hub](https://huggingface.co/models), [TensorFlow Hub](https://tfhub.dev)
- **Tutorials**: [HuggingFace Course](https://huggingface.co/course), [FastAI NLP](https://docs.fast.ai)
- **Books**: "Speech and Language Processing" (Jurafsky), "Natural Language Processing with Transformers"
