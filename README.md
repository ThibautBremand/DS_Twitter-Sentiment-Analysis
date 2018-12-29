# Data Science - Twitter Sentiment Analysis
A Data Science project on Python Jupyter Notebook.  
Prediction of the **polarity of tweets using a Naive Bayes classification model**, based on **emoticons**.  

## 1. Context and objective
The main inspiration for this notebook comes from this research paper from students of the Stanford University (**Sentiment410 project**).  
https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf  

For the moment, I use the same dataset as them, which is available under : http://help.sentiment140.com/for-students

The objective is being able to predict the polarity of a tweet, based on a classification model buildt from a large dataset of classified tweets. **The tweets are classified depending upon the emoticons they contain**, which means that we don't use a referential of positive and negative words, but we build a ML model using *'polarity by emoticon classification'*.  

## 2. Training Dataset (Calculation of polarity and Data Wrangling)  

### 2.1 Calculation of the tweets polarity of the training set  
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

### 2.2 Data Wrangling  
The Data Wrangling method I chose ot used is detailed under : https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn  
Each tweet is prepared as such :  
- The tweets are converted into lowercase
- The punctuation is removed
- The tweets are normalized using a word stemming method : the Porter Stemmer algorithm is one of the most popular one  
- The tweets are converted into occurences using CountVectorizer, and weighted using Term Frequency Inverse Document Frequency  

## 3. Model training  
I used the Multinomial Naive Bayes Classifier algorithm, which is well-suited for text classification problems.  
With the current dataset, using this algorithm, We obtain 76% accuracy.

## 4. What's next  
One main objective is to keep gather tweets through the Twitter API and calculate the polarity of these using emoticons. These would be then used by our model to keep learning and improve it's accuracy.  


#### Author : Thibaut BREMAND  
- thibaut.bremand [at] gmail.com
- https://github.com/ThibautBremand

#### Sources :  
- https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf : The original research paper which inspered me
- http://help.sentiment140.com  : The original training dataset  
- https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/ : Explanation of the methodology. 
- https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn : More details about the Naive Bayes classification
