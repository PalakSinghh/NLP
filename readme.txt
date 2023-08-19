This code is an implementation of sentiment analysis on news data using a Support Vector Classifier (SVC) and a logistic regression model. The analysis involves several steps, including data preprocessing, feature extraction, and model training. Here's a summary of the code:

Importing Libraries and Loading Data:

Libraries like pandas, numpy, matplotlib, and nltk are imported.
A CSV file named "News Dataset.csv" is read using pandas to load the news data.
Data Exploration:

Basic exploratory data analysis is performed using pandas functions such as head(), shape, columns, info(), and isnull().sum().
Text Data Preprocessing:

The text data from columns 2 to 27 of the DataFrame is combined into a single list named text_data.
A new DataFrame named data is created with two columns: "Text" (contains the preprocessed text data) and "Label" (contains the labels from the original dataset).
Visualization of Data Distribution:

Matplotlib and Plotly are used to create bar and pie charts to visualize the distribution of label values in the dataset.
Text Preprocessing and Tokenization:

NLTK's libraries are used for text preprocessing, including lowercasing, tokenization, lemmatization, and removing stopwords.
The preprocessed data is stored in the corpus list.
Building Word Frequencies:

The frequency of each word in the dataset is calculated and stored in the freqs dictionary.
Feature Extraction:

CountVectorizer is used to convert the text data into a matrix of token counts.
The data is split into training and testing sets using the train_test_split function.
Model Training - Support Vector Classifier (SVC):

An SVM model from the sklearn library is used for training.
The model is trained on the training data and evaluated using the testing data.
Classification report and accuracy score are printed for evaluation.
Applying Logistic Regression (Additional Model):

An additional approach using logistic regression is presented.
A function extract_features is defined to extract features from the text data using the freqs dictionary.
The sigmoid function and gradient descent are implemented for logistic regression.
The model is trained using gradient descent and the weights are updated iteratively.
Prediction using Logistic Regression Model:

A function predict_text is defined to predict sentiment using the logistic regression model and extracted features.
The function is tested on example texts.
Overall, this code demonstrates how to preprocess text data, extract features, and train both an SVM and a logistic regression model for sentiment analysis on news data. It also includes visualizations to understand the distribution of sentiment labels in the dataset. Additionally, the logistic regression implementation provides an alternative approach for sentiment analysis.




