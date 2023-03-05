from django.shortcuts import render

from django.http import HttpResponse
from .forms import RegisterForm

import regex as re
import requests
import json  
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import emoji
from underthesea import word_tokenize


def index(request):
    if request.method == 'POST':
        comments = []
        s = requests.Session()
        response = HttpResponse()
                
        apiKey = "AIzaSyAmR-BGlhakyXqj6UYAS6iyRT5RElM5Kz4"
        
        maxResults = 1000
        videoId = request.POST['link'].split("v=")[-1]
        pageToken = ""
        
        for i in range(0, 100):
            res = s.get(f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&key={apiKey}&videoId={videoId}&maxResults={maxResults}&pageToken={pageToken}")
            jsonOb = json.loads(res.text)
            comments = comments + get_comments(jsonOb)
            try:
                pageToken = jsonOb['nextPageToken']
            except:
                break
        with open('comments.txt', 'w', encoding='utf-8') as wf:
            for cmt in comments:
                wf.write(cmt)
                wf.write('\n')
  
    registerForm = RegisterForm()
    return render(request, 'Sentiment/index.html',  {'form': registerForm})

def get_comments(jsonOb):
    comments = []
    for item in jsonOb['items']:
        comments.append(item['snippet']['topLevelComment']['snippet']['textOriginal'])
    return comments
    
def result(request):
    # Đọc comments đã crawl
    comments = open("comments.txt", "r", encoding='utf-8')
    
    # Load model đã train
    crawl_model = joblib.load('crawl_model.sav')
    
    # Load vectorizer
    vectorizer = joblib.load('vectorizer.sav')
    
    # Chuyển comments đã crawl sang dataFrame
    df_cmt = pd.DataFrame(comments, columns = ['Sentence'])
    
    my_test = df_cmt.copy()
    
    #Preprocessing Text
    vietnamese_stopwords=[
    'bị'
    ,'bởi'
    ,'cả'
    ,'các'
    ,'cái'
    ,'cần'
    ,'càng'
    ,'chỉ'
    ,'chiếc'
    ,'cho'
    ,'chứ'
    ,'chưa'
    ,'chuyện'
    ,'có'
    ,'có_thể'
    ,'cứ'
    ,'của'
    ,'cùng'
    ,'cũng'
    ,'đã'
    ,'đang'
    ,'đây'
    ,'để'
    ,'đến_nỗi'
    ,'đều'
    ,'điều'
    ,'do'
    ,'đó'
    ,'được'
    ,'dưới'
    ,'gì'
    ,'khi'
    ,'không'
    ,'là'
    ,'lại'
    ,'lên'
    ,'lúc'
    ,'mà'
    ,'mỗi'
    ,'một_cách'
    ,'này'
    ,'nên'
    ,'nếu'
    ,'ngay'
    ,'nhiều'
    ,'như'
    ,'nhưng'
    ,'những'
    ,'nơi'
    ,'nữa'
    ,'phải'
    ,'qua'
    ,'ra'
    ,'rằng'
    ,'rằng'
    ,'rất'
    ,'rất'
    ,'rồi'
    ,'sau'
    ,'sẽ'
    ,'so'
    ,'sự'
    ,'tại'
    ,'theo'
    ,'thì'
    ,'trên'
    ,'trước'
    ,'từ'
    ,'từng'
    ,'và'
    ,'vẫn'
    ,'vào'
    ,'vậy'
    ,'vì'
    ,'việc'
    ,'với'
    ,'vừa']
    
    def remove_emoji(text):
        return emoji.get_emoji_regexp().sub("", text)

    def basic_preprocessing(document):
        # đưa về lower
        document = document.lower()
        # xóa các ký tự không cần thiết
        document = re.sub(r'[^\w\s]',' ',document)
        # xóa khoảng trắng thừa
        document = re.sub(r'\s+', ' ', document).strip()

        return document

    def remove_stopwords(mess):
        STOPWORDS = vietnamese_stopwords
        # Check characters to see if they are in punctuation
        nopunc = [char for char in mess if char not in string.punctuation]
        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
        # Now just remove any stopwords
        return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
    
    my_test.Sentence = my_test.Sentence.apply(remove_emoji)

    word_segment = lambda x: word_tokenize(x, format="text")

    my_test.Sentence = my_test.Sentence.apply(word_segment)

    my_test.Sentence = my_test.Sentence.apply(remove_stopwords)

    my_test.Sentence = my_test.Sentence.apply(basic_preprocessing)
    
    X_my_test = vectorizer.transform(my_test.Sentence)
    
    col = [c for c in my_test.columns if 'Unnamed' not in c]
    
    input = my_test[col]
    y_pred = crawl_model.predict(X_my_test)
    df_cmt.insert(1, "Emotion", y_pred)
    # test_1_report = classification_report(y_my_test, y_pred)

    df_cmt.loc[df_cmt["Emotion"] == 0, "Emotion"] = "Anger"
    df_cmt.loc[df_cmt["Emotion"] == 1, "Emotion"] = "Disgust"
    df_cmt.loc[df_cmt["Emotion"] == 2, "Emotion"] = "Enjoyment"
    df_cmt.loc[df_cmt["Emotion"] == 3, "Emotion"] = "Fear"
    df_cmt.loc[df_cmt["Emotion"] == 4, "Emotion"] = "Other"
    df_cmt.loc[df_cmt["Emotion"] == 5, "Emotion"] = "Sadness"
    df_cmt.loc[df_cmt["Emotion"] == 6, "Emotion"] = "Surprise"
    
    json_records = df_cmt.reset_index().to_json(orient ='records')
    
    #Thong ke
    emotionTotal = pd.DataFrame(df_cmt.Emotion.value_counts(), columns = ['Emotion'])
    
    json_result = emotionTotal.reset_index().to_json(orient ='records')
    
    total = []
    total = json.loads(json_result)
    
    data = []
    data = json.loads(json_records)
    context = {'d': data, 't': total}
  
    return render(request, 'Sentiment/table.html', context)