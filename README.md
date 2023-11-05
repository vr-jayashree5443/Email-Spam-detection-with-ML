# Email-Spam-detection-with-ML
Here's a detailed explanation of each part of the code:

1. **Importing Libraries**:
   - Importing necessary Python libraries, including `numpy` and `pandas` for data manipulation, `matplotlib.pyplot` for data visualization, `seaborn` for creating visually appealing plots, and modules from `sklearn` for machine learning.

2. **Loading the Dataset**:
   - Loading the email spam dataset from a CSV file using `pandas`. The dataset is read into a DataFrame.

3. **Data Preprocessing**:
   - Extracting the email text data ('v2') and labels ('v1') from the DataFrame.
   - Converting the labels 'ham' and 'spam' into binary values (0 for 'ham' and 1 for 'spam').
   - Converting the email text data to a list of strings to prepare it for feature extraction.

4. **Feature Extraction (CountVectorizer)**:
   - Using the `CountVectorizer` from `sklearn` to convert the text data into a matrix of token counts. This process converts the text into numerical features, representing the frequency of each word in the emails.

5. **TF-IDF Transformation (TfidfTransformer)**:
   - Applying the TF-IDF (Term Frequency-Inverse Document Frequency) transformation to the token count matrix. This step assigns weights to words based on their importance in the document and the corpus, making it more suitable for machine learning.

6. **Splitting the Data**:
   - Splitting the dataset into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. This helps ensure the model generalizes well to new data.

7. **Training the Classifier (Multinomial Naive Bayes)**:
   - Choosing the Multinomial Naive Bayes classifier, a common choice for text classification tasks. Training the classifier using the training data allows it to learn patterns and relationships within the email text and their corresponding labels.

8. **Making Predictions**:
   - Using the trained model to make predictions on the testing set. The model assigns labels (0 for 'ham' and 1 for 'spam') to the emails in the test set.

9. **Model Evaluation (Accuracy)**:
   - Calculating the accuracy of the model by comparing the predicted labels with the actual labels in the testing set. Accuracy measures the percentage of correctly classified emails.

10. **Confusion Matrix**:

    ![Figure_1](https://github.com/vr-jayashree5443/Email-Spam-detection-with-ML/assets/128161257/49e48088-fd2e-494c-a2ca-20c96aaed0d6)

    - Generating a confusion matrix to understand the model's performance in terms of true positives, true negatives, false positives, and false negatives. This provides a deeper insight into the model's performance.

12. **Visualization (Seaborn)**:
    - Visualizing the confusion matrix using the `seaborn` library. The heatmap provides a clear representation of the model's performance in distinguishing between 'ham' and 'spam' emails.

13. **Classification Report**:

    ![Screenshot 2023-11-05 114823](https://github.com/vr-jayashree5443/Email-Spam-detection-with-ML/assets/128161257/fd1eeec1-5541-404b-9de0-747fccb78c6a)

    - Generating a classification report, which provides a detailed summary of various classification metrics such as precision, recall, F1-score, and support for both 'ham' and 'spam' classes. This report provides a comprehensive view of the model's performance.

15. **Sample Email Predictions**:
    - Defining a list of sample emails to test the model. These emails are used to check how well the model predicts the labels for new, unseen data.

16. **Predicting Sample Emails**:

    ![Screenshot 2023-11-05 114848](https://github.com/vr-jayashree5443/Email-Spam-detection-with-ML/assets/128161257/f71dee70-4ac9-400f-96ab-da73f3244dbc)

    - Preprocessing and predicting the labels for the sample emails using the trained model. The model assigns labels ('ham' or 'spam') to these sample emails.

18. **Displaying Sample Email Predictions**:
    - Printing the model's predictions for the sample emails, showing whether the model classifies each email as 'ham' or 'spam.'

The code follows a structured workflow, from data preprocessing to model evaluation and sample email predictions, to build an email spam detector using machine learning.
