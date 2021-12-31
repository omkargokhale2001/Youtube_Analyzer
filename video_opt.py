import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import calendar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LinearRegression
from nltk.stem import WordNetLemmatizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import preprocessor as p
import re
from mlxtend.regressor import StackingRegressor
import nltk
from nltk.stem import PorterStemmer


def duration_to_time(duration):
    times = duration.split('M')
    if len(times) >= 2 and times[1] != '':
        for i in times[0]:
            if i == 'H':
                hrs_min = times[0].split('H')
                hrs = int(hrs_min[0][2:])*60
                minutes = int(hrs_min[1])
                modify_time = hrs + minutes + float(times[1][:-1])/60
                return modify_time
        modify_time = float(times[0][2:]) + float(times[1][:-1])/60
    else:
        if 'S' in times[0]:
            modify_time = float(times[0][2:-1])/60
        else:
            modify_time = float(times[0][2:])
    return round(modify_time, 2)


def day_of_week(date):
    time_ = date.split('T')
    date = time_[0]
    date = datetime.strptime(date, '%Y-%m-%d').weekday()
    return date


def find_time(time):
    time_ = time.split('T')
    time_fin = time_[1]
    time_fin = int(time_fin[0:2])*60 + int(time_fin[3:5])
    return time_fin


def find_day(day):
    day_list = day.split('T')
    day_fin = day_list[0]
    day_fin = (int(day_fin[5:7])-1)*30 + int(day_fin[8:10])
    return day_fin


def remove_stop(title,stop_words):
    stop_words = set(stopwords.words('english'))
    new_sent = ""
    words = word_tokenize(title)
    for i in words:
        if i not in stop_words:
            new_sent = new_sent + i
            new_sent = new_sent + " "
    return new_sent


def clean_tweet(text):
    text1 = re.sub(r'@[A-Za-z0-9]+', '', text)
    text2 = re.sub(r'#', '', text1)
    hashtags = re.findall(r'#[A-Za-z0-9]+', text)
    hashtags = [i[1:] for i in hashtags]
    text3 = re.sub(r'RT[\s]+', '',text2)
    text4 = re.sub(r'https?:\/\/\S+', '', text3)
    text5 = re.sub(r'http?:\/\/\S+', '', text4)
    text6 = re.sub(r'\n', '', text5)
    return text6


def remove_emoticons(tweet):
    tweet = p.clean(tweet)
    return tweet


def stem_words(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])