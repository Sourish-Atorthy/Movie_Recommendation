# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:26:49 2022

@author: arya, indrani & sourish
"""

# import numpy as np
import pandas as pd
import ast as a
import streamlit as st
import re
import csv
import os
import requests
import random
# import seaborn as sns
# import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LinearRegression,Lasso,Ridge
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

st.title('MovieMania.com')
st.caption('Get the best movie recommendations')

# kaggle datasets download -d tmdb/tmdb-movie-metadata
credits=pd.read_csv('tmdb_5000_credits.csv')
movies=pd.read_csv('tmdb_5000_movies.csv')

movies=movies.merge(credits,on='title')
movies=movies[['genres','id','keywords','overview','title','cast','crew','revenue']]
#movies.isnull().sum()
movies=movies.dropna()
#movies.isnull().sum()
#movies.duplicated().sum()

def change(obj):
    L=[]
    for i in a.literal_eval(obj): #To convert string to list
        L.append(i['name'])
    return L

movies['genres']=movies['genres'].apply(change)
movies['keywords']=movies['keywords'].apply(change)

def cast_fn3(obj):
    L=[]
    counter=0
    for i in a.literal_eval(obj): #To convert string to list
        if counter !=3:
            counter=counter+1
            L.append(i['name'])
        else:
            break
    return L


movies['cast']=movies['cast'].apply(cast_fn3)

def director(obj):
    L=[]
    for i in a.literal_eval(obj): #To convert string to list
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L

movies['crew']=movies['crew'].apply(director)

#Converting to list
movies['overview']=movies['overview'].apply(lambda x:x.split())
movies['genres']=movies['genres'].apply(lambda x: [i.replace(" ","") for i in x]) #Concating Science and Fiction together, to avoid tags confusion

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new_df=movies[['id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x)) #To convert the list to string 
new_df['tags']=new_df['tags'].apply(lambda x: x.lower())

lt = WordNetLemmatizer()
def lemment(text):
    y=[]
    for i in text.split():
        y.append(lt.lemmatize(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(lemment)

#Text Vectorization using Bag of words methods
cv= CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
# st.text(cv.get_feature_names())

#Cosine Distance
similarity=cosine_similarity(vectors)
#similarity[0]

def listToString(s):
    # initialize an empty string
    str1 = " "
    # return string 
    return (str1.join(s))

# text=cv.get_feature_names()
names_list = cv.get_feature_names()
random.shuffle(names_list)
wc_text=listToString(names_list)

wordcloud = WordCloud(collocations = False, background_color = 'white').generate(wc_text)
# Display the generated Word Cloud
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

#Now we have to create a function to find the similar movies for each movies, but will first store the index numbers before sorting
#sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6] #lambda function is used to make the 2nd column as key for sorting, enumerate converts list to tuples

def fetch_poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=3e671f025e2bf97b296b3d27e159a83e&language=en-US'.format(movie_id))
    data=response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def recommend(movie):    
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]    

    st.subheader("Recommended Movies Are:")
    
    for i in movies_list:
        st.subheader(new_df.iloc[i[0]].title)
        st.image(fetch_poster(new_df.iloc[i[0]].id))


st.header("Recommendations")

if st.checkbox("Select if you want to see movie recommendations"):
    label="Enter a movie name for seeing recommendations"
    options=new_df['title']
    movie_name = st.selectbox(label, options, index=0)
    recommend(movie_name)



st.header("")
st.header("Feedback Form")

if st.checkbox("Select if you want to give a feedback"):       
    rating=st.number_input("Rate our site",min_value=1.0,max_value=10.0,step=1.0)
    st.text("Please write to us with an honest feedback")

    email=st.text_input("Enter your email", placeholder="john.doe@gmail.com")
    email_regex = re.compile(r"[^@]+@[^@]+\.[^@]+")
    if email_regex.match(email) == None:
        st.caption("Please enter a valid email!")
    
    feedback=st.text_area("Enter your feedback here")
    pic=None
    fb_picture=st.checkbox("Select if you are open to sending your picture")
    if fb_picture:
        pic=st.camera_input("Your picture", disabled=False)
    
    if st.button("Submit Feedback"):
        st.caption("Thank you for your feedback "+email+"!")
    
        with open('feedbacks.csv', 'w', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
        
            if os.stat('feedbacks.csv').st_size == 0:
                header = ['Email', 'Rating', 'Feedback', 'Picture']
                writer.writerow(header)
            
            # write a row to the csv file
            data = [email,rating,feedback,pic]
            writer.writerow(data)

st.header("")
st.header("")
st.text("Made by Arya Dasgupta, Indrani Chakraborty & Sourish Atorthy")