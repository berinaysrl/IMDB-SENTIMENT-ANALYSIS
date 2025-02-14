# IMDB SENTIMENT ANALYSIS PROJECT

In this project, I analyzed the sentiment of IMDB movie reviews using a combination of Natural Language Processing (NLP) techniques and a machine learning model. The goal was to predict whether a movie review had a positive or negative sentiment based on the text content.



## Table of Contents

**[Project Overview](#Project-Overview)** 

**[Data Source](#Data-Source)**

**[Installation & Running the Project](#Installation-&-Running-the-Project)**

**[Approach](#Approach)**  

**[Findings](#Findings)**  

## Project Overview

The purpose of this project was to classify IMDB movie reviews into positive or negative sentiments using machine learning. The workflow consisted of:

- Preprocessing: Cleaning and preparing text data for analysis.
 
- Exploratory Data Analysis (EDA): Visualizing patterns in the data to gain insights about the data and model's performance.
 
- Model Training: Building a deep learning model to predict review sentiment.
 
-	Evaluation: Measuring model performance through metrics like accuracy, confusion matrices, and classification reports. 


## Data Source

You may access the dataset used in this project through this [link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

The dataset has two basic characteristics: 
1. review: contains the reviews in IMDB.
2. sentiment: positive or negative sentiment annotations for each review.
  

## Installation & Running the Project

1. Install the required packages:
   
```
   pip install tensorflow numpy pandas matplotlib seaborn nltk beautifulsoup4 wordcloud scikit-learn
```

2. Clone the repository

```
   git clone https://github.com/yourusername/imdb-sentiment-analysis.git
```

3. Navigate to the project directory

```
   cd imdb-sentiment-analysis
```

4. do not forget to update the csv file on run.py, modify this like correspondingly

```
   csv_path = "/Users/berinayzumrasariel/Desktop/imdb sentiment project/data/IMDB Dataset.csv"
```
5. run his command

```
   python run.py
```

## Approach

1. Preprocessing
   Data is cleaned from html tags, special characters that are not alphanumeric, and stopwords. Then the stemming applied to each word. This was necessary for the model to focus on only the meaningful text and it improved the efficiency. Stemming was chosen over lemmatization because of its simplicity and therefore computation speed.

2. Exploratory Data Analysis
   I used some visualizations to give the user a sense of what we are dealing with. I thought it would be nice to see which words are mostly appearing on positive and negative reviews, so a function is added to generate word clouds which displayed the most used words in positive and negative reviews.

3. Building  and Evaluating the Model
   I chose to build a simple neural netword with an embedding layer, pooling and dropout layers. This way, it was possible to have over 80% accuracy without needing to use SOTA models like BERT. To evaluate the model, I used accuracy, precision, recall, F1-score, and confusion matrices. 

## Findings

The model achieved an 87.69% accuracy on the test set.
	- 	Positive reviews were predicted with 91% precision and 84% recall, indicating a slightly lower ability to correctly identify all positive reviews compared to precision.
	- 	Negative reviews showed 85% precision and 92% recall, suggesting a stronger ability to identify true negatives but a slightly higher false positive rate.

Potential Limitations  

 1.  Overfitting to Training Data
	-	Training loss consistently decreased while validation loss plateaued or increased slightly after several epochs, suggesting mild overfitting.  
 
 3.  Imbalanced Class Performance
	- Precision and recall for positive and negative classes showed variability, with negative reviews having stronger recall but lower precision.

 5.  Preprocessing Choices
	-	Stemming may have removed meaningful word variations, potentially impacting the nuanced understanding of review content.


   

    

