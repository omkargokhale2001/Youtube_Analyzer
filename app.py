import pymongo
from flask import Flask, render_template, request, session, redirect, url_for
from googleapiclient.discovery import build
import requests
import json
from datetime import datetime
import pandas as pd
import plotly
import plotly.graph_objs as go
from flask_pymongo import PyMongo
import yaml
import dns
from sklearn.ensemble import StackingRegressor
from nltk.tag import pos_tag
import time

from video_opt import *

app = Flask(__name__)
app.secret_key = "oogabooga"

mongodb_client = pymongo.MongoClient("mongodb+srv://admin:a6ae99d3a163@cluster0.1e2er.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = mongodb_client['youtube_cred_db']


def plot_chart(df, param):
    df_new = df.sort_values(by='Date')
    data = [
        go.Line(
            x=df_new['Date'],
            y=df_new[param]
        )
    ]

    graphjson = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphjson


def to_date(date):
    year = date[0:4]
    day = date[8:10]
    month = date[5:7]
    month_dict = {'01': 'January',
             '02': 'February',
             '03': 'March',
             '04': 'April',
             '05': 'May',
             '06': 'June',
             '07': 'July',
             '08': 'August',
             '09': 'September',
             '10': 'October',
             '11': 'November',
             '12': 'December'	}
    month = month_dict[month]
    return day + " " + month + " " +year


def get_channel_stats_by_id(key, channel_id):
    url = "https://youtube.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&id="+str(channel_id)+"&key="+str(key)
    print(url)
    data = requests.get(url)
    data = json.loads(data.text)
    result = {
        "name": data['items'][0]['snippet']['title'],
        "subs": data['items'][0]['statistics']['subscriberCount'],
        "channel_start": data['items'][0]['snippet']['publishedAt'],
        "uploads": data['items'][0]['contentDetails']['relatedPlaylists']['uploads'],
        "prof_pic": data['items'][0]['snippet']['thumbnails']['default']['url']
    }
    return result


def get_channel_by_name(key, name):
    url = "https://youtube.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&forUsername="+str(name)+"&key="+str(key)
    data = requests.get(url)
    data = json.loads(data.text)
    return data

def get_channel_id(name, key):
    url = "https://youtube.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q="+str(name)+"&key="+str(key)
    print(url)
    data = requests.get(url)
    print(data)
    data = json.loads(data.text)
    return data['items'][0]['snippet']['channelId']

def get_duration(date):
    date = date[:10]
    date_format = "%Y-%m-%d"
    a = datetime.strptime(date, date_format)
    b = datetime.strptime(str(datetime.now())[:10], date_format)
    delta = b - a
    years = int(delta.days/365)
    days = delta.days % 365
    months = int(days/12)
    days = days % 12
    return "Creating content since {} years, {} months, {} days".format(years, months, days)


def generate_model(df):
    df['Video_name'] = df['Video_name'].apply(lambda x: x.lower())
    df['Duration'] = df['Duration'].apply(lambda x: duration_to_time(x))
    df['Day'] = df['Date'].apply(lambda x: day_of_week(x))
    df['Time'] = df['Date'].apply(lambda x: find_time(x))
    df['Days'] = df['Date'].apply(lambda x: find_day(x))
    df['Video_name'] = df['Video_name'].str.lower()
    stop_words = set(stopwords.words('english'))
    df['Video_name'] = df['Video_name'].apply(lambda x: remove_stop(x, stop_words))
    df['Video_name'] = df['Video_name'].apply(lambda x: clean_tweet(x))
    df['Video_name'] = df['Video_name'].apply(lambda x: remove_emoticons(x))
    df['Video_name'] = df['Video_name'].apply(lambda x: stem_words(x))
    df['ratio'] = (df['Likes'].astype('int32') / df['Views'].astype('int32')) * 100
    vect = TfidfVectorizer(ngram_range=[1, 2]).fit(df['Video_name'])
    x_train = vect.transform(df['Video_name'])
    x_train_2 = df[['Duration', 'Day', 'Time', 'Days']]
    x_train_main = pd.merge(pd.DataFrame(x_train.toarray()), x_train_2, left_index=True, right_index=True)
    y_train = df['Views']
    reg_svr_1 = SVR().fit(x_train, y_train)
    reg_svr_2 = SVR().fit(x_train_2, y_train)
    reg_svr = SVR()
    svr_reg = StackingRegressor(regressors=[reg_svr_1, reg_svr_2],
                                meta_regressor=reg_svr)
    svr_reg.fit(x_train_main, y_train)
    return svr_reg, vect


def get_optimal_params(df):
    nltk.download('averaged_perceptron_tagger')
    words_imp = []
    df_sorted = df.sort_values(by='Views')
    lst_names = list(df['Video_name'])[15:]
    for i in range(len(lst_names)):
        tagged_sent = pos_tag(lst_names[i].split())
        for word in tagged_sent:
            if word[1] == 'NN' and len(word[0]) >= 1:
                words_imp.append(word[0])
    duration = list(df_sorted['Duration'])[15:]
    duration = [int(duration_to_time(x)) for x in duration]
    return words_imp, round((sum(duration)/len(duration)), 2)


@app.route('/', methods=['GET', 'POST'])
def search_name():
    if request.method == 'POST':
        if len(session) != 0:
            session.clear()
        channel_name = request.form['channel_name']
        print("Channel name is:", channel_name)
        session['channel_name'] = channel_name
        f = open("api_key.txt", 'r')
        key = str(f.read())
        f.close()
        print(key)
        cur = db.get_collection("channel_id").find({"channel_name": channel_name.replace(" ", "").lower()})
        channel_id = ''
        for data in cur:
            channel_id = data['channel_id']
        print("Channel id is:", channel_id)
        if channel_id:
            session['channel_id'] = channel_id
            stats = get_channel_stats_by_id(key, channel_id)
            session['uploads'] = stats['uploads']
            days = get_duration(stats['channel_start'])
            session['statistics'] = stats
        else:
            channel_id = get_channel_id(channel_name, key)
            print(channel_id)
            session['channel_id'] = channel_id
            stats = get_channel_stats_by_id(key, channel_id)
            session['statistics'] = stats
            session['uploads'] = stats['uploads']
            days = get_duration(stats['channel_start'])
            db.get_collection("channel_id").insert_one({"channel_name": channel_name.replace(" ", "").lower(), "channel_id": channel_id})
        channel_id = session['channel_id']
        uploads_id = session['uploads']
        f = open("api_key.txt", 'r')
        key = str(f.read())
        f.close()
        url = 'https://youtube.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=20&playlistId=' + str(
            uploads_id) + '&key=' + str(key)
        data = requests.get(url)
        data = json.loads(data.text)
        dates = []
        names = []
        likes = []
        views = []
        comments = []
        hashtags = []
        duration = []
        for i in range(20):
            title = data['items'][i]['snippet']['title']
            names.append(title)
            video_id = data['items'][i]['snippet']['resourceId']['videoId']
            dates.append(data['items'][i]['snippet']['publishedAt'])
            description = data['items'][i]['snippet']['description'].split('#')
            hashtags.append(description[1:])
            url = 'https://youtube.googleapis.com/youtube/v3/videos?part=contentDetails%2Cstatistics&id=' + str(
                video_id) + '&key=' + str(key)
            video_data = requests.get(url)
            video_data = json.loads(video_data.text)
            likes.append(video_data['items'][0]['statistics']['likeCount'])
            views.append(video_data['items'][0]['statistics']['viewCount'])
            comments.append(video_data['items'][0]['statistics']['commentCount'])
            duration.append(video_data['items'][0]['contentDetails']['duration'])
        length = 20
        df = pd.DataFrame({"Date": dates, "Video_name": names, "Views": views, "Likes": likes, "Comments": comments,
                           "Hashtags": hashtags, "Duration": duration})
        dict_obj = df.to_dict('list')
        session['data'] = dict_obj
        return render_template('index.html', name=stats['name'], subs=stats['subs'], date=days, id=channel_id,
                               profile=stats['prof_pic'])
    return render_template('index.html')


@app.route('/video_projections', methods=['GET'])
def video_proj():
    data = session['data']
    df = pd.DataFrame(data)
    names = list(df['Video_name'])
    names.reverse()
    likes = list(df['Likes'])
    likes.reverse()
    comments = list(df['Comments'])
    comments.reverse()
    dates = [to_date(x) for x in list(df['Date'])]
    dates.reverse()
    views = list(df['Views'])
    views.reverse()
    per_change = ["-"]
    for i in range(1, len(views)):
        init_views = int(views[i-1])
        cur_views = int(views[i])
        percent = ((cur_views-init_views)/init_views)*100
        per_change.append(str(round(percent, 1)) + "%")
    length = 20
    bar1 = plot_chart(df, 'Views')
    bar2 = plot_chart(df, 'Likes')
    bar3 = plot_chart(df, 'Comments')
    return render_template('channel_growth.html', names=names, views=views, likes=likes, comments=comments, dates=dates, percent=per_change, length=length, plot1=bar1, plot2=bar2, plot3=bar3)


@app.route('/video_optimization', methods=['GET', 'POST'])
def video_opt():
    data = session['data']
    df = pd.DataFrame(data)
    print(df)
    names = list(df['Video_name'])
    likes = list(df['Likes'])
    comments = list(df['Comments'])
    dates = list(df['Date'])
    dates = [to_date(x) for x in dates]
    length = 20
    views = list(df['Views'])
    words, ideal_time = get_optimal_params(df)
    if request.method == 'POST':
        model, vector = generate_model(df)
        video_name = request.form['video_name']
        video_duration = request.form['video_duration']
        day = time.strptime(request.form['day'], "%A").tm_wday
        days = request.form['days']
        days = str(days)
        days = int(days[5:7])*30 + int(days[8:10])
        time_of_upload = request.form['time']
        minutes = int(str(time_of_upload)[0:2])*60 + int(str(time_of_upload)[3:5])
        x_test_fin = pd.DataFrame({'Duration': [video_duration], 'Day': [day], 'Time': [minutes], 'Days': [days]})
        x = [video_name]
        x = vector.transform(x)
        x_test_1 = pd.DataFrame(x.toarray())
        x_test_2 = x_test_fin
        x_test_final = pd.merge(x_test_1, x_test_2, left_index=True, right_index=True)
        predictions = round(float(str(model.predict(x_test_final))[1:-1]), 2)
        print(x_test_fin)
        return render_template('video_opt.html', names=names, likes=likes, comments=comments, length=length, views=views, dates=dates, time=ideal_time, words=words, predictions=predictions)
    return render_template('video_opt.html', names=names, likes=likes, comments=comments, length=length, views=views, dates=dates, time=ideal_time, words=words)


@app.route('/best_videos')
def find_best():
    df = pd.DataFrame(session['data'])
    names = list(df['Video_name'])
    likes = list(df['Likes'])
    comments = list(df['Comments'])
    dates = list(df['Date'])
    views = list(df['Views'])
    dates = [to_date(x) for x in dates]
    length = 20
    ratios = []
    for i in range(len(df)):
        ratio = round((int(df['Likes'].iloc[i]) / int(df['Views'].iloc[i]))*100, 2)
        ratios.append(ratio)
    df['ratio'] = ratios
    index = df['ratio'].astype('float64').idxmax()
    ratio = list(df.iloc[index])
    ratio[1] = to_date(ratio[1])
    print(ratio)
    index = df['Views'].astype('float64').idxmax()
    best_video_views = list(df.iloc[index])
    best_video_views[1] = to_date(best_video_views[1])
    print(df)
    return render_template('best_videos.html', stats=best_video_views, ratio=ratio, names=names, views=views, likes=likes, comments=comments, dates=dates, length=length)


# main driver function
if __name__ == '__main__':
    app.run()

