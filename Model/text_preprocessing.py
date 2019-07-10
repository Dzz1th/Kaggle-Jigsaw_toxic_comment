import pandas as  pd 
import numpy as np
import gensim
from gensim import corpora, models
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
import operator

stopwords = gensim.parsing.preprocessing.STOPWORDS

EMBED_SIZE = 300
MAX_FEATURES = 10000 #the number of unique words
MAXLEN = 220 #max lenght of commented text

    
def make_dictionary(data, text_column = 'comment_text', no_below = 10, no_above = 0.5, keep_n = 7500):
    dictionary = gensim.corpora.Dictionary(data[text_column])
    dictionary.filter_extremes(no_below = no_below, no_above = no_above, keep_n = keep_n)
    return dictionary

def make_bow(data, dictionary, text_column = 'comment_text'):
    bow_corpus = [dictionary.doc2bow(doc) for doc in data[text_column]]
    return bow_corpus

def make_tfidf(data , bow_corpus):
    tfidf = gensim.models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf

def build_LDA_model(corpus_tfidf, id2word, num_topics = 20, passes = 2, workers = 3):
    lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics = num_topics, id2word = id2word, passes=passes, workers = workers)
    return lda_model

    


def delete_stopwords(text, stop_words = stopwords):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stopwords and len(token) > 2:
            result.append(token)
    return result


def make_vocabulary(texts):
    sentenses = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentenses:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def load_embedding(file_path):
    def get_coef(word , *arr):
        return word, np.asarray(arr, dtype='float32')
    embedding_index =dict(get_coef(*o.split(" ")) for o in open(file_path , encoding='latin'))
    return embedding_index

def embedding_matrix(word_index, embeddings_index):
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    EMBED_SIZE = all_embs.shape[1]
    nb_words = min(MAX_FEATURES, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBED_SIZE))
    for word, i in tqdm(word_index.items()):
        if i >= MAX_FEATURES:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def check_coverage(vocab , embedding_index):
    known_words = {}
    unknown_words = {}
    num_known_words = 0
    num_unknown_words  = 0
    for word in tqdm(vocab.keys()):
        try:
            known_words[word] = embedding_index[word]
            num_known_words += vocab[word]
        except KeyError:
            unknown_words[word] = vocab[word]
            num_unknown_words += vocab[word]
            pass
    print('Found embedding for {:.2%} of vocabulary'.format(len(known_words) /len(vocab)))
    print('Found embedding for {:.2%} of text'.format(num_known_words /(num_known_words + num_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    return unknown_words

def add_lower_in_emb(embedding_matrix , vocab):
    count = 0
    for word in tqdm(vocab.keys()):
        if word in embedding_matrix and word.lower() not in embedding_matrix:
            embedding_matrix[word.lower()] = embedding_matrix[word]
            count += 1
    print('{} word added'.format(count))

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
    "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
    "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
    "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",
    "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 
    "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
    "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 
    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is",
    "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", 
    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", 
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", 
    "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", 
    "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", 
    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", 
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", 
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not","won't've": "will not have", 
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 
    "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are",
    "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 
    'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization',
    'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
    'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 
    'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 
    'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',
    'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018',
    'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what',
    'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization',
     'demonetisation': 'demonetization'}

def known_contractions(embed):
    known=[]
    for contr in tqdm(contraction_mapping):
        if contr in embed:
            known.append(contr)
    return known

def clean_contractions(text, mapping = contraction_mapping):
    '''
    input: current text, contraction mappings
    output: modify the comments to use the base form from contraction mapping
    '''
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

punct_mapping = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping += '©^®` <→°€™› ♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'

def unknown_punct(embed, punct):
    '''
    input: current text, contraction mappings
    output: unknown punctuation
    '''
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown

puncts = {"‘": "'", "´": "'", "°": "", "€": "e", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '…': ' '}

def clean_special_chars(text, punct, mapping):
    '''
    input: current text, punctuations, punctuation mapping
    output: cleaned text
    '''
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ') 
    return text


TEST_PATH = 'c:/Data/data/test.csv'
TRAIN_PATH = 'c:/Data/data/train.csv'

test = pd.read_csv(TEST_PATH , index_col='id')
train = pd.read_csv(TRAIN_PATH , index_col='id')
EMB_PATH = 'c:/Data/data/glove.txt'

emb_index = load_embedding(EMB_PATH)
vocab = make_vocabulary(train['comment_text'])
emb_matrix = embedding_matrix(vocab, emb_index)

check_coverage(vocab, emb_index)
add_lower_in_emb(emb_matrix , vocab)
train['comment_text'] = train['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
test['comment_text'] = test['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

train['comment_text'] = train['comment_text'].apply(lambda x: clean_special_chars(x, punct_mapping, puncts))
test['comment_text'] = test['comment_text'].apply(lambda x: clean_special_chars(x, punct_mapping, puncts))

df = pd.concat([train ,test], sort=False)
vocab = make_vocabulary(df['comment_text'])
print("Check coverage after punctuation replacement")
oov_glove = check_coverage(vocab, emb_index)

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(train))
train = tokenizer.texts_to_sequences(train)
test = tokenizer.texts_to_sequences(test)