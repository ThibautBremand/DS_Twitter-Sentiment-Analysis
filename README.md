# Data Science - Twitter Sentiment Analysis
A Data Science project on Python Jupyter Notebook.  
Prediction of the **polarity of tweets using a Naive Bayes classification model**, based on **emoticons**.  

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

### 2.2 Data Wrangling  
The Data Wrangling method I chose to use is detailed under : https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn  
Each tweet is prepared as such :  
- The tweets are converted into lowercase
- The @username are replaced with *USERNAME* tokens
- The URLs are replaced with *URL* tokens
- Any letter occurring more than two times in a row is replaced with two occurrences
- The punctuation is removed
- The tweets are normalized using a word stemming method : the Porter Stemmer algorithm is one of the most popular one  
- The tweets are converted into occurences using CountVectorizer, and weighted using Term Frequency Inverse Document Frequency  

## 3. Model training  
I used the Multinomial Naive Bayes Classifier algorithm, which is well-suited for text classification problems.  
With the current dataset, using this algorithm, **we obtain 77% accuracy**.

## 4. What's next  
One main objective is to automatically keep gathering tweets through the Twitter API and calculate the polarity of these using emoticons. These would be then used by our model to keep learning and improve it's accuracy.  
A notebook explaining how to gather data from the API is available in this repo, the aim is to automatically use this method to gather data continuously.  

Another objective is to adapt this model to other examples, for example French which is my mother tongue.  


## Author : Thibaut BREMAND  
- thibaut.bremand [at] gmail.com
- https://github.com/ThibautBremand

### Sources :  
- https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf : The original research paper which inspered me
- http://help.sentiment140.com  : The original training dataset  
- https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/ : Explanation of the methodology   
- https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn : More details about the Naive Bayes classification
