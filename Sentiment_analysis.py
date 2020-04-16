import numpy as np 
import pandas as pd 
import re  
import nltk  
import matplotlib.pyplot as plt
nltk.download('stopwords')  
from nltk.corpus import stopwords  
 
tweets = pd.read_csv("https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")
 
tweets.head()
 
tweets.shape
 
plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0]) 
print(plot_size[1])

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size 

tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
 
tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])

airline_sentiment = tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')

import seaborn as sns

sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=tweets)

#sns.countplot(x='airline_sentiment', data=tweets)
 
#sns.countplot(x='airline', data=tweets[0:20])

#sns.countplot(x='airline', hue="airline_sentiment", data=tweets)
 
 
X = tweets.iloc[:, 10].values  
y = tweets.iloc[:, 1].values
 
 
processed_tweets = []
 
for tweet in range(0, len(X)):  
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))
 
    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
 
    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
 
    # Substituting multiple spaces with single space
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
 
    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
 
    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()
 
    processed_tweets.append(processed_tweet)
    
    
 
from sklearn.feature_extraction.text import TfidfVectorizer  
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(processed_tweets).toarray()
 
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# using KNN classifier method


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
 
 
predictions = classifier.predict(X_test)
 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("The Confusion Matrix is:\n")
print(confusion_matrix(y_test,predictions))  
print("\n The Classification Report is:\n")
print(classification_report(y_test,predictions))
print("\n The accuracy Score is:\n")  
print(accuracy_score(y_test, predictions))



# using Random Forest classifier method 
 
from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)  
text_classifier.fit(X_train, y_train)
 
 
predictions = text_classifier.predict(X_test)
 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("The Confusion Matrix is:\n")
print(confusion_matrix(y_test,predictions))  
print("\n The Classification Report is:\n")
print(classification_report(y_test,predictions))
print("\n The accuracy Score is:\n")  
print(accuracy_score(y_test, predictions))


 