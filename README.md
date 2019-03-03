# Data Science - Twitter Sentiment Analysis
A Data Science project in Python, using Jupyter Notebook.  
Prediction of the **polarity of tweets using multiple classification model**, based on **emoticons**.  

One Notebook is about **building the dataset using the Twitter API**, the other one is about **training the Machine Learning model to predict the polarity of new tweets**.

## 1. Context and objective
Twitter is such a powerful tool used simultaneously by thousands of people to express themselves about a lot of varied topics. With only one query, you can find out what the world think about the subject you want, and you can reply to everyone. It represents a huge source of data when you want to analyze what the world thinks about a specific subject : a product you just launched, some worldwide concerns, the new record of your favorite band, etc...  

The idea is to build a Machine Learning model that will predict whether a tweet has been written in a positive or negative mind. With an accurate model, you will then be able to check if the tweets about, let's say, the new product your company just launched, are positive or negative. You will be able to distinguish a trend. 

The main inspiration for this notebook comes from this research paper from students of the Stanford University (**Sentiment410 project**).  
https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf  

For the moment, I use the same dataset as them, which is available under : http://help.sentiment140.com/for-students

The objective is being able to predict the polarity of a tweet, based on a classification model built from a large dataset of classified tweets. **The tweets are classified depending upon the emoticons they contain**, which means that we won't use a sentiment lexicon based approach (referential of positive and negative words), but we build a ML model using **polarity by emoticon classification**.  

## 2. Training Dataset 

### 2.1 Calculation of the Tweets polarity  
The method to calculate the polarity of the tweets in the training dataset is the same as the one used in the Sentiment410 : **usage of emoticons** to determine wether a tweet is positive or negative  
- A large amount of tweets is collected through the Twitter public API.
- If the tweet contains a positive emoticon, it will be categorized as positive. With a negative emoticon, it will be caterogized as such.  
- Only the positive or negative tweets (meaning they countain at least one emoticon) are used for the training set. The emoticons are then removed from the textual tweet to not bias the classification model. 

| Emoticons mapped to :) | Emoticons mapped to :( |
|---|---|
| :) | :( |
| :-) | :-( |
| : ) | : ( |
| :D |
| =) |

[Example of negative tweet used in the Sentiment410 training set](https://twitter.com/Karoli/status/1467811193)  

### 2.2 Dataset analysis

#### 2.2.1 - Repartition of the polarity of tweets within the dataset 

![Count](images/1-Counts.PNG?raw=true)

The dataset contains 800k positive tweets and 800k negative tweets. Both tweet polarities are equally represented within our dataset.  

#### 2.2.2 - Wordclouds of main words for both polarities

Positive tweets wordcloud :  
![PositiveWordcloud](images/2-Wordcloud1.png?raw=true)

Negative tweets wordcloud :  
![NegativeWordcloud](images/3-Wordcloud2.png?raw=true)


### 2.3 Data Wrangling  
The Data Wrangling method I chose to use is detailed under : https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn  
Each tweet is prepared as such :  
- The tweets are converted into lowercase
- The @username are replaced with *USERNAME* tokens
- The URLs are replaced with *URL* tokens
- Any letter occurring more than two times in a row is replaced with two occurrences
- The punctuation is removed
- The tweets are normalized using a word stemming method : the Porter Stemmer algorithm is one of the most popular one  

In order to perform machine learning on text documents, we first need to turn the text content into numerical feature vectors :  
- The tweets are tokenized : the text is converted into arrays of tokens. I chose to use **Unigrams** as tokens, so each token will contain one word.  
- The tokens are converted into word occurences using **CountVectorizer**  
- These occurences are weighted and normalized using **Term Frequency Inverse Document Frequency (TFIDF)**. This technique intends to reflect how important a word is in the collection of tweets, by pounding it : Stop words impact will be limited, rare words will get special weight.  

a) Five rows of the original dataset :

| sentiment_score | id         | date                           | query    | author          | tweet                                               |
|-----------------|------------|--------------------------------|----------|-----------------|-----------------------------------------------------|
| 0               | 1467810369 | Mon   Apr 06 22:19:45 PDT 2009 | NO_QUERY | _TheSpecialOne_ | @switchfoot   http://twitpic.com/2y1zl - Awww, t... |
| 0               | 1467810672 | Mon   Apr 06 22:19:49 PDT 2009 | NO_QUERY | scotthamilton   | is   upset that he can't update his Facebook by ... |
| 0               | 1467810917 | Mon   Apr 06 22:19:53 PDT 2009 | NO_QUERY | mattycus        | @Kenichan   I dived many times for the ball. Man... |
| 0               | 1467811184 | Mon   Apr 06 22:19:57 PDT 2009 | NO_QUERY | ElleCTF         | my   whole body feels itchy and like its on fire    |
| 0               | 1467811193 | Mon   Apr 06 22:19:57 PDT 2009 | NO_QUERY | Karoli          | @nationwideclass   no, it's not behaving at all.... |

b) The same rows with normalized, tokenized, and stemmed tweets  

| sentiment_score | id         | date                           | query    | author          | tweet                                               |
|-----------------|------------|--------------------------------|----------|-----------------|-----------------------------------------------------|
| 0               | 1467810369 | Mon   Apr 06 22:19:45 PDT 2009 | NO_QUERY | _TheSpecialOne_ | [usernam,   url, aww, that, a, bummer, you, shou... |
| 0               | 1467810672 | Mon   Apr 06 22:19:49 PDT 2009 | NO_QUERY | scotthamilton   | [is,   upset, that, he, cant, updat, hi, faceboo... |
| 0               | 1467810917 | Mon   Apr 06 22:19:53 PDT 2009 | NO_QUERY | mattycus        | [usernam,   i, dive, mani, time, for, the, ball,... |
| 0               | 1467811184 | Mon   Apr 06 22:19:57 PDT 2009 | NO_QUERY | ElleCTF         | [my,   whole, bodi, feel, itchi, and, like, it, ... |
| 0               | 1467811193 | Mon   Apr 06 22:19:57 PDT 2009 | NO_QUERY | Karoli          | [usernam,   no, it, not, behav, at, all, im, mad... |

## 3. Models training  
I used the following algorithms which benefit from a very fast cumputation time. I also know that Multinomial Naive Bayes Classifier algorithm is well-suited for text classification problems, so I decided to try it out first.  

Here is the accuracy for each algorithm, considering that the tokens are made of **Unigrams**.  

| Algorithm           | Accuracy on training set | Accuracy on test set |
|---------------------|--------------------------|----------------------|
| Naive Bayes         | 0.81                     | 0.77                 |
| SVM (Linear)        | 0.42                     | 0.35                 |
| SVM (SGD)           | 0.78                     | 0.78                 |
| Logistic Regression | 0.81                     | 0.80                 |

All the algorithms, except for the Linear SVM, benefit from a good accuracy (~80% on test set).  

## 4. What's next  
One main objective is to automatically keep gathering tweets through the Twitter API and calculate the polarity of these using emoticons. These would be then used by our model to keep learning and improve it's accuracy.  
A notebook explaining how to gather data from the API is available in this repo, the aim is to automatically use this method to gather data continuously.  

Another objective is to adapt this model to other languages, for example French which is my mother tongue.  


## Author : Thibaut BREMAND  
- thibaut.bremand [at] gmail.com
- https://github.com/ThibautBremand

### Sources :  
- https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf : The original research paper which inspired me
- http://help.sentiment140.com : The original training dataset
- https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/ : Explanation of the methodology.
- https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn : More details about the Naive Bayes classification
- https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2 : Algorithms comparison using a simple classification problem
- https://medium.com/@sangha_deb/naive-bayes-vs-logistic-regression-a319b07a5d4c : Naive Bayes and Logistic Regression algorithms comparison
