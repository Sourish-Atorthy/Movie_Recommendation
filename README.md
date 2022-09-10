# Movie Recommendation System

## Background
A movie recommendation system, also known as a movie recommender system, uses machine learning (ML) to predict or filter users' film preferences based on their prior decisions and actions. The current period is referred as the "age of abundance." There can be thousands of options available for any given product. For example: social networking, online shopping, streaming videos, and so forth. Recommender systems assist in personalizing a platform and assisting users in finding content they enjoy.
Just like YouTube recommends content, or Facebook recommends new friends, Amazon recommends similar products while we’re browsing, movie recommendation system predicts similar movies based on the input movie name what the user has entered. All these recommendations are made possible by the implementation of recommender systems.

We require specialized recommender systems in order to truly improve the user experience through customized recommendations. From a business perspective, user engagement is higher for the more relevant products they discover on the site. Increased platform revenue is an outcome of this. Various sources claim that as much as 35–40% of the revenue of internet behemoths comes from only referrals.

For this purpose, we have made a movie recommendation system in which user will enter the movie name of his/her choice, then based on the movie’s cast, crew, genre, keywords used in overview similar other movie’s names will be recommended to the user. With this user will be easily able to identify the similar movies he/ she wants to watch. It will save a lot of time and a good movie recommendation system will increase user satisfaction. Additionally, we have kept a feedback feature in our app, which will get us valuable feedbacks and rating of the app from our users. By analysing the feedbacks given by each user, we be able to update/add features in our movie recommendation system.

## Data Source
We have collected the datasets from Kaggle.

## Tools Used
1.	MS Excel - Contains the data file
2.	Jupyter Notebook - Used for creating the web application
3.	Spyder - Used for developing integrated app using python language
4.	Streamlit – Used to create and host the front end of the app
5.	GitHub – Used as a hosting service for development and version control.

## Important Techniques Used
We have used various techniques and algorithms to get to a good result. The various techniques used are as follows:

### Cosine Similarity
Cosine similarity is a measure of similarity, often used to measure document similarity in text analysis. We have used the Cosine Similarity from Sklearn, as the metric to compute the similarity between two movies. This is a statistic used to assess how similar two items are. It calculates the cosine of the angle formed by two vectors that are projected onto a multidimensional space.

We use the below formula to compute the cosine similarity:

Similarity = (A.B) / (||A||.||B||)

where A and B are vectors: 
•	A.B is dot product of A and B: It is computed as sum of element-wise product of A and B.
•	||A|| is L2 norm of A: It is computed as square root of the sum of squares of elements of the vector A.
The output value is between 0 and 1. The normalised dot product of the input samples X and Y is how the Python Cosine Similarity calculates similarity. To determine the cos for the two vectors in the count matrix, we have used the sklearn cosine similarity function. 0 means no similarity, whereas 1 means that both the items are 100% similar.

### Text Vectorization
Text Vectorization is the process of converting text into numerical representation. In programming, a vector is a data structure that is similar to a list or an array. For the purpose of input representation, it is simply a succession of values, with the number of values representing the vector’s “dimensionality.” Vector representations contain information about the qualities of an input object. They offer a uniform format that computers can easily process.

There are some popular methods to accomplish text vectorization. They are: 
1.	Binary Term Frequency. 
2.	Bag of Words (BoW) 
3.	Term Frequency. (L1) 
4.	Normalized Term Frequency.

### Bag of Words
There are many state-of-art approaches to extract features from the text data. The simplest and known method is the Bag-Of-Words representation. It’s an algorithm that transforms the text into fixed-length vectors. This is possible by counting the number of times the word is present in a document. The word occurrences allow to compare different documents and evaluate their similarities for applications, such as search, document classification, and topic modelling.

A BoW vector has the length of the entire vocabulary — that is, the set of unique words in the corpus. The vector’s values represent the frequency with which each word appears in a given text passage.

