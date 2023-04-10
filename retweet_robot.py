import tweepy
import time
import json
import requests
import pandas as pd
from hazm import Normalizer,Stemmer,Lemmatizer,WordTokenizer,stopwords_list
import numpy as np
import pickle
import string
from tensorflow.keras.models import load_model
from tokens import telegram_token , ap_chanel_id , bearer_token , consumer_key , consumer_secret , access_token , access_token_secret
import os
dirname = os.path.dirname(__file__)

print(dirname)
def chek_fohsh(inp):
    fohsh=[
            "جمهوری_اسلامی",
            "توله",
            "طویله",
            "سرنگونی",
            "چس",
            "#نه_به_ولایت_فقیه",
            "#نه_به_جمهوری_اسلامی",
            "#نه_به_جمهوری‌اسلامی",
            "#قیام_تا_سرنگونی",
            "گوز",
            "لجن",
            "کثافت",
            "بی شرف",
            "بیشعور",
            "گوه",
            "کون",
            "کیری",
            "کسکش",
            "بی ناموس",
            "سگ پدر",
            "پدرسگ",
            "شاش",
            "ریدن",
            "ریدی",
            "کونی",
            "دیوس",
            "انی",
            "گهی",
            "بی پدر",
            "مادرسگ",
            "بی ناموس",
            "جنده",
            "گایدی",
            "گایدن",
            "کیر",
            "کیروکس",
            "عمتو",
            "خفه شو",
            "خفه",
            "خفه خون",
            "مرض داری",
            "مرضداری",
            "گردن دراز",
            "خری",
            "گاوی",
            "اسبی",
            "سگی",
            "حیوانی",
            "دهنتوببند",
            "انگل",
            "آشغال",
            "خرفت",
            "پپه",
            "خنگ",
            "دکل",
            "دله",
            "قرتی",
            "گوزو",
            "کونده",
            "کون ده",
            "گاگول",
            "ابله",
            "گنده گوز",
            "کس",
            "کله کیری",
            "گشاد",
            "دخترقرتی",
            "خواهرجنده",
            "مادرجنده",
            "لخت",
            "بخورش",
            "بپرسرش",
            "بپرروش",
            "بیابخورش",
            "میخوریش",
            "بمال",
            "دیوس خان",
            "زرنزن",
            "زنشو",
            "زنتو",
            "زن جنده",
            "بکنمت",
            "بکن",
            "بکن توش",
            "بکنش",
            "لز",
            "سکس",
            "سکسی",
            "ساک",
            "ساک بزن",
            "پورن",
            "سکسیی",
            "کونن",
            "کیرر",
            "جاکش",
            "انی",
            "بدبخت",
            "خایه",
            "خایه مال",
            "خایه خور",
            "ممه",
            "ممه خور",
            "دخترجنده",
            "خارکسده",
            "کس ننت",
            "کیردوس",
            "مادرکونی",
            "خارکسده",
            "خارکس ده",
            "کیروکس",
            "کس و کیر",
            "زنا",
            "زنازاده",
            "ولدزنا",
            "ملنگ",
            "سادیسمی",
            "فاحشه",
            "خانم جنده",
            "فاحشه خانم",
            "سیکتیر",
            "سسکی",
            "کس خیس",
            "حشری",
            "گاییدن",
            "بکارت",
            "داف",
            "بچه کونی",
            "کسشعر",
            "سرکیر",
            "سوراخ کون",
            "حشری شدن",
            "کس کردن",
            "کس دادن",
            "بکن بکن",
            "شق کردن",
            "کس لیسیدن",
            "آب کیر",
            "جاکش",
            "جلق زدن",
            "جنده خانه",
            "شهوتی",
            "عن",
            "قس",
            "کردن",
            "کردنی",
            "کس لیس",
            "کس کش",
            "کوس",
            "کیرمکیدن",
            "لاکونی",
            "پستان",
            "آلت",
            "آلت تناسلی",
            "نرکده",
            "مالوندن",
            "سولاخ",
            "جنسی",
            "دوجنسه",
            "سگ تو روحت",
            "بی غیرت",
            "نعشه",
            "بی عفت",
            "مادرقهوه",
            "پلشت",
            "پریود",
            "کله کیری",
            "کیرناز",
            "پشمام",
            "لختی",
            "کسکیر",
            "دوست دختر",
            "دوست پسر",
            "کونشو",
            "دول",
            "شنگول",
            "کیردراز",
            "داف ناز",
            "سکسیم",
            "کوص",
            "ساکونی",
            "کون گنده",
            "سکسی باش",
            "کسخل",
            "صیغه ای",
            "گوش دراز",
            "درازگوش",
            "توله سگ",
            "خز",
            "ماچ",
            "ماچ کردنی",
            "اسکل",
            "هیز",
            "بیناموس",
            "اوسکل",
            "بی آبرو",
            "لاشی",
            "لاش گوشت",
            "باسن",
            "جکس",
            "سگ صفت",
            "کصکش",
            "مشروب",
            "عرق خور",
            "سکس چت",
            "جوون",
            "سرخور",
            "کلفت",
            "حشر",
            "لاس",
            "زارت",
            "خر",
            "گاو",
            "اسب",
            "گوسفند",
            "الاق",
            "الاغ",
            "احمق",
            "بی شعور",
            "حرومزاده",
            "کونی",
            "گه",
            "مادر جنده",
            "کث",
            "کص",
            "پسون",
            "خارکسّه",
            "دهن گاییده",
            "دهن سرویس",
            "پدر سگ",
            "پدر سوخته",
            "پدر صلواتی",
            "لامصب",
            "زنیکه",
            "مرتیکه",
            "مردیکه",
            "بی خایه",
            "عوضی",
            "اسگل",
            "اوسکل",
            "اوسگل",
            "اوصگل",
            "اوصکل",
            "دیوث",
            "دیوص",
            "قرمصاق",
            "قرمساق",
            "غرمساق",
            "غرمصاق",
            "فیلم سوپر",
            "چاقال",
            "چاغال",
            "چس خور",
            "کس خور",
            "کس خل",
            "کوس خور",
            "کوس خل",
            "کص لیس",
            "کث لیس",
            "کس لیس",
            "کوص لیس",
            "کوث لیس",
            "کوس لیس",
            "کون سوراخ",
            "سوراخ کون",
            "شورت",
            "ریدم",
            "#ریدم_تو_اسلام",
            "کون پنیر"
    ]
    for item in fohsh:
        if item in inp:
            return True
    return False



#function Tweet to dict
def tweet_to_dict(tweet):
    d = {}
    d["id"] = tweet.id
    d["text"] = tweet.text
    d["author_id"] = tweet.author_id
    d["public_metrics"] = tweet.public_metrics
    return d

#ارسال توییت‌ها داخل تلگرام
def send_to_telegram(matn):
#TODO
    base_url = "https://api.telegram.org/bot"+telegram_token+"/sendMessage"


#TODO
    parameters = {"chat_id":ap_chanel_id , "text": matn}
    r = requests.get(base_url, params=parameters)


#TODO
auth = tweepy.OAuthHandler( consumer_key  , consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth , wait_on_rate_limit = True)

#TODO
client = tweepy.Client(bearer_token = bearer_token,
                       consumer_key = consumer_key,
                       consumer_secret = consumer_secret,
                       access_token = access_token,
                       access_token_secret = access_token_secret,
                       wait_on_rate_limit = True)


list_vaje = [
    'مدارس',
    'تربیتی','تعلیم تربیت','آ.پ.','آ.پ','آموزش و پرورش','دانشگاه فرهنگیان','تربیت معلم','دانش سرا','ماده 28','پرورشی','خرید خدمات آموزشی',
    'رتبه بندی', 'رتبه بندی', 'رتبه_بندی', 'غیر انتفاعی', 'مدرسه', 'دبستان', 'مهدکودک', 'پیش دبستانی', 'دبیرستان',
    'علوم تربیتی', 'برنامه درسی', 'نظام آموزشی', 'کلاس' ,'school' ,'education']
query= ""
list_query = [list_vaje[0]+ " -is:retweet lang:fa" ]
i=1
j=0
while i <len(list_vaje):
    temp=list_query[j]
    list_query[j] += " OR "+list_vaje[i] +" -is:retweet lang:fa"

    if 492 < len(list_query[j]):
        list_query[j]=temp
        list_query.append(list_vaje[i] +" -is:retweet lang:fa")
        j+=1
    i+=1
adress=dirname
def retweet_ap(rooz_ghabl = 1):
    #store date of today in last day and the day before yesterday in ld in date (ISO 8601) format
    d = time.strftime("%Y-%m-%d", time.localtime(time.time() -rooz_ghabl* 86400))+"T00:00:00.000Z"
    ld = time.strftime("%Y-%m-%d", time.localtime(time.time() - (rooz_ghabl+1)*86400)) +"T00:00:00.000Z"


    tweet_of_today = set({})
    page=0
    for x in list_query:

        page=0
        search=client.search_recent_tweets(x+" -is:retweet lang:fa" ,start_time = ld , end_time = d ,user_auth=False , tweet_fields= ["public_metrics","text","author_id","id"] )
        while True:
            try:
                page+=1
                print("page : ",page)
                for tweet in search.data:
                    tweet_of_today.add(tweet)
                search=client.search_recent_tweets(x+" -is:retweet lang:fa",next_token=search.meta['next_token'] ,start_time = ld , end_time = d ,user_auth=False , tweet_fields= ["public_metrics","text","author_id","id"] )

            except:
                break


    tweet_fohsh = set({})
    tweet_normal = set({})
    dtweet_fohsh = []
    dtweet_normal = []
    for tweet in tweet_of_today:
        t_fohsh = len (dtweet_fohsh)
        t_normal = len (dtweet_normal)
        if chek_fohsh(tweet.text):
            tweet_fohsh.add(tweet)
            if len(tweet_fohsh) != t_fohsh:

                dtweet_fohsh.append(tweet_to_dict(tweet))
        else:

            tweet_normal.add(tweet)

            if len(tweet_normal) != t_normal:
                dtweet_normal.append(tweet_to_dict(tweet))
    with open(adress+"/data/"+d[:-14]+'fohsh.json', 'w') as f:
        #indent=4 is for readability
        json.dump(dtweet_fohsh, f, indent=4)
    with open(adress+"/data/"+d[:-14]+'normal.json', 'w') as f:
        #indent=4 is for readability
        json.dump(dtweet_normal, f, indent=4)

    #read from d+normal.json to dataframe
    df = pd.read_json(adress+"/data/"+d[:-14]+'normal.json' , orient='records',)
    df['retweet_count'] = df['public_metrics'].apply(lambda x: x['retweet_count'])
    df['like_count'] = df['public_metrics'].apply(lambda x: x['like_count'])
    df['reply_count'] = df['public_metrics'].apply(lambda x: x['reply_count'])
    df['quote_count'] = df['public_metrics'].apply(lambda x: x['quote_count'])

    df["emtiyaz"]= df['retweet_count']*5 + df['like_count'] + df['reply_count']*2 + df['quote_count']*3 + [x[0]*700 for x in text_to_emtiyaz(df["text"])]


    #sort df by emtiyaz
    df.sort_values(by=['emtiyaz'], ascending=False, inplace=True)
    #retweet 10 tweets with highest emtiyaz and more than 10 emtiyaz
    ret= df.head(10)
    for x in ret[ret["emtiyaz"]>10]["id"]:
        print(x)
        try:
            api.retweet(x)
        except:
            pass

    i=0
    for x in ret[ret["emtiyaz"]>10]["text"]:
        i+=1
        send_to_telegram(x+"\n\n"+str(i)+"\n"+str(d))



#________________AI____________AAAAAAIIIIII
#_______________AAII___________AAAAAAIIIIII
#______________AAAIII______________AAII
#_____________AAA  III_____________AAII
#____________AAA    III____________AAII
#___________AAAAAAIIIIII___________AAII
#__________AAAAAAAIIIIIII__________AAII
#_________AAA          III_________AAII
#________AAA            III________AAII
#_______AAA              III___AAAAAAIIIIII
#______AAA                III__AAAAAAIIIIII

#import model.h5 as model whith tensorflow
model = load_model(dirname+'/pickle_x/model_2022_01_25.h5')


f = open(dirname+'/pickle_x/tkn', 'rb')
tkn = pickle.load(f)
f.close()

def pad_sequences(sequences, maxlen=None, padding = 'pre'):
    #create a array of zeros with len(sequences) rows and maxlen columns
    #if maxlen is None, maxlen = max(len(s) for s in sequences)
    if maxlen is None:
        maxlen = max(len(s) for s in sequences)
    khorooji = np.array([np.zeros(maxlen) for s in sequences])
    for i, s in enumerate(sequences):
        if s!=[]:
            #if len(s) < maxlen:
            #    if pading == 'pre':
            #        khorooji[i, -len(s):] = s
            #    else:
            #        khorooji[i, :len(s)] = s
            #else:

            khorooji[i, -len(s):] = s
    return khorooji


normalizer = Normalizer()
table = str.maketrans('', '', string.punctuation+'\n'+"1234567890۱۲۳۴۵۶۷۸۹۰qwertyuiopasdfghjklzxcvbnmQWERTYUIOPLKJHGFDSAZXCVBNM،," )
stemmer = Stemmer()
lemmatizer = Lemmatizer()
tokenizer = WordTokenizer(join_verb_parts=False, replace_IDs=True, replace_numbers=True)
def text_to_list(text):
    text = text.translate(table)
    text = normalizer.normalize(text)
    text = tokenizer.tokenize(text)
    #remove stop words
    text = [word for word in text if word not in stopwords_list()]
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)

    return text

def array_text_to_padded_docs(sotoon_text):
    seq = tkn.texts_to_sequences(sotoon_text.apply(text_to_list))

    padded_docs = pad_sequences(seq, padding = 'pre', maxlen= 70)
    return padded_docs


def text_to_emtiyaz(text):
    return model.predict(array_text_to_padded_docs(text))

