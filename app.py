# importing libraries
import pymongo
import pandas as pd
import re
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.feature_extraction.text import TfidfVectorizer 
from flask import Flask, render_template, request, Markup

# instantiate the stemmer
ps = PorterStemmer()

# creating connection to the mongo DB
mongoUrl = "mongodb+srv://prod_readuser:eDHou1mlatGmDWOZ@mystudypath-cluster.vxfex.mongodb.net/institution-prod-db?retryWrites=true&w=majority"
client = pymongo.MongoClient(mongoUrl)
mydb = client['institution-prod-db']
col = mydb["course"]
courses = col.find({}, {'name':1, 'status':'Published', 'schoolName':1, 'schoolStatus':"Published"})

# collecting and storing the course info in a dictionary for ease of access
courseDf = {'courseID':[], 'name': [], 'courseStatus': [], 'school': []}
for course in courses:
    courseDf['courseID'].append(str(course.get('_id')))
    courseDf['name'].append(course.get('name'))
    courseDf['courseStatus'].append(course.get('status'))
    courseDf['school'].append(course.get('schoolName'))
paveCourses = pd.DataFrame(courseDf)

# data cleaning
paveCourses['length'] = paveCourses['name'].str.split().apply(len)

def findBracket(value):
    try:
        index = value.index('(')
        return int(index)
    except ValueError:
        return "None"
paveCourses['bracketIndex'] = paveCourses['name'].apply(findBracket)
paveCourses['cleanName'] = paveCourses[['name', 'bracketIndex']].apply(lambda x: x[0][:x[1]] if x[1]!='None' else x[0], axis=1)

def findHyphen(value):
    try:
        index = value.index('-')
        return int(index)
    except ValueError:
        return "None"
paveCourses['hyphenIndex'] = paveCourses['name'].apply(findHyphen)
paveCourses['cleanName2'] = paveCourses[['cleanName', 'hyphenIndex']].apply(lambda x: x[0][:x[1]] if x[1]!='None' else x[0], axis=1)

paveCourses.drop(['name', 'length', 'bracketIndex', 'cleanName', 'hyphenIndex'], axis=1, inplace=True)
paveCourses.rename(columns={'cleanName2':'courseName'}, inplace=True)

def cleanCourses(course):
    course = course.lower()
    course = re.sub(r'[^a-z]', ' ', course)
    course = course.strip()
    course = nltk.word_tokenize(course)
    course = [word for word in course if word not in stopwords.words('english')]
    course = [ps.stem(word) for word in course]
    course = ' '.join(course).strip()
    return course
paveCourses['stemCourses'] = paveCourses['courseName'].apply(cleanCourses)


# building the recommender
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
vectorizer.fit_transform(paveCourses['stemCourses'])

arr = vectorizer.fit_transform(paveCourses['stemCourses']).toarray()
def courseVectorize(course):
    course = pd.Series(cleanCourses(course))
    courseVec = vectorizer.transform(course).toarray()
    return courseVec

# function to carry out the cosine similarity algorithm and return a dataframe showing the cosine similarity calculated and their respective index from the data.
def cosine(data, test):
    result = cos_sim(data, test)
    result_df = pd.DataFrame(result, columns=['similarityScore'])
    result_df.sort_values('similarityScore', ascending=False, inplace=True)
    topIndex = result_df
    return topIndex

def recommend(text):
    test = courseVectorize(text)
    if text!='':
        similarDf = cosine(arr, test)
        similarIndex = similarDf.index[:]
        result = ""
        for index in similarIndex:
            result+=f"Simimlarity Score: {'===='*5}> {similarDf.loc[index, 'similarityScore']}\n"
            result+="CourseID: "+paveCourses.loc[index, 'courseID']
            result+='\n'
            result+="Course Status: "+paveCourses.loc[index, 'courseStatus']
            result+='\n'
            result+="Course Name: "+paveCourses.loc[index, 'courseName'] 
            result+='\n'
            result+="School Name: "+paveCourses.loc[index,'school']
            result+='\n\n'
        return result
    else:
        print('\n\nPlease type the course you want to search for :)')


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user = request.form["course"]
    result = recommend(user).replace('\n', '<br>')
    return render_template('index.html', prediction_text=Markup(result))
    #return render_template('index.html', prediction_text=user)

    
if __name__=='__main__':
    app.run(debug=True)
