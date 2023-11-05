
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("D:/JJ/Oasis Infobyte/4_Email Spam Detection with Machine Learning/archive (3)/spam.csv")  
X = data['v2']
y = data['v1']

y = y.map({'ham': 0, 'spam': 1})
X = X.tolist()

count_vectorizer = CountVectorizer()
X_counts = count_vectorizer.fit_transform(X)


tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

sample_emails = ["Go until jurong point, crazy.. Available only in bugis n great world la e buffet...","Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's","I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]
sample_emails_counts = count_vectorizer.transform(sample_emails)
sample_emails_tfidf = tfidf_transformer.transform(sample_emails_counts)

sample_predictions = classifier.predict(sample_emails_tfidf)
sample_predictions = ['spam' if pred == 1 else 'ham' for pred in sample_predictions]
print("Sample Email Predictions:", sample_predictions)
