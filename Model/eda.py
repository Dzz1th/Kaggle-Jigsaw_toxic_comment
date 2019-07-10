import gc
import os
import warnings
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from wordcloud import WordCloud, STOPWORDS
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
from keras.preprocessing.text import Tokenizer
import text_preprocessing

stopwords = set(STOPWORDS)

count = 0

def make_feature_plot(data, features, title = 'distribution of feature in dataset'):
    global count
    plt.figure(count, figsize=(12, 6))
    count += 1
    plt.title(title)
    for feature in features:
        sns.distplot(data.loc[~data[feature].isnull() , feature] , kde = True , hist = False , bins=20 , label = feature)
    plt.xlabel('')
    plt.legend()
    plt.show()

def make_target_plot(data, target ='target', title = 'distribution of target in dataset'):
    global count
    plt.figure(count, figsize = (12 , 6))
    count += 1
    plt.title(title)
    sns.distplot(data[target], bins = 20, hist = False , kde = True, label = target)
    plt.legend()
    plt.show()

def make_count_plot(data , feature, size = 1):
    f , ax = plt.subplot(1 , 1 , figsize =(4*size , 4))
    total = float(len(data))
    g = sns.countplot(data[feature] , order = data[feature].value_counts().index[:20] , palette='Set3')
    title = '{} count plot'.format(feature)
    g.set_title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()

def show_wordcloud(data , title=None , stop_words = stopwords, max_words = 50):
    global count
    wordcloud = WordCloud(
        background_color='white',
        stopwords = stop_words,
        max_words = max_words,
        max_font_size = 40,
        scale = 5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(count, figsize = (10 , 10))
    count += 1
    plt.axis('off')
    if(title):
        fig.suptitle(title, fontsize = 20)
        fig.subplot_adjust(top=2.3)
        
    plt.imshow(wordcloud)
    plt.show()

def eval_topic_using_lda(ldamodel, bow_corpus, texts):
    topics_df = pd.DataFrame()
    for i , row in enumerate(ldamodel[bow_corpus]):
        row = sorted(row , key = lambda x: (x[1]) , reverse = True)
        for j , (topic_num, prop_topic) in enumerate(row):
            if j==0:
                wp = ldamodel.show_topic
                topic_keywords = ", ".join([word for word, prop in wp])
                topics_df = topics_df.append(pd.Series([int(topic_num) , round(prop_topic,4), topic_keywords]) , ignore_index=True)
            else:
                break
    topics_df = pd.concat([topics_df , texts] , axis = 1)
    topics_df.reset_index()
    return topics_df

def make_box_plot(data , feature_x , feature_y):
    sns.boxplot(y = feature_y, x = feature_x, data= data, orient='h')

def make_corr_heatmap(data):
    corr_matr = data.corr()
    sns.heatmap(corr_matr)




