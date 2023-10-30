mport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df= pd.read_csv('spam.csv')

df.isnull().sum()

df['Spam'] = np.where(df['Category'] == 'spam', 1, 0) 
plt.figure(figsize=(6, 4))
df['Spam'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Spam (1) and Non-Spam (0)')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

X = df['Message']
y = df['Spam']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

svm_classifier = SVC()
rf_classifier = RandomForestClassifier()
nb_classifier = MultinomialNB()
lr_classifier = LogisticRegression()
gb_classifier = GradientBoostingClassifier()

svm_classifier.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)
nb_classifier.fit(X_train, y_train)
lr_classifier.fit(X_train, y_train)
gb_classifier.fit(X_train, y_train)


ensemble_classifier = VotingClassifier(estimators=[
    ('SVM', svm_classifier),
    ('Random Forest', rf_classifier),
    ('Multinomial Naive Bayes', nb_classifier),
    ('Logistic Regression', lr_classifier),
    ('Gradient Boosting', gb_classifier)
], voting='hard')

ensemble_classifier.fit(X_train, y_train)

y_pred_svm = svm_classifier.predict(X_test)
y_pred_rf = rf_classifier.predict(X_test)
y_pred_nb = nb_classifier.predict(X_test)
y_pred_lr = lr_classifier.predict(X_test)
y_pred_gb = gb_classifier.predict(X_test)
y_pred_ensemble = ensemble_classifier.predict(X_test)

models = {
        'SVM': y_pred_svm,
        'Random Forest': y_pred_rf,
        'Multinomial Naive Bayes': y_pred_nb,
        'Logistic Regression': y_pred_lr,
        'Gradient Boosting': y_pred_gb,
        'Ensemble': y_pred_ensemble
    }

acc_svm = accuracy_score(y_test, y_pred_svm)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_nb = accuracy_score(y_test, y_pred_nb)
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_gb = accuracy_score(y_test, y_pred_gb)
acc_ensemble = accuracy_score(y_test, y_pred_ensemble)

print("SVM Accuracy:", acc_svm)
print("Random Forest Accuracy:", acc_rf)
print("Multinomial Naive Bayes Accuracy:", acc_nb)
print("Logistic Regression Accuracy:", acc_lr)
print("Gradient Boosting Accuracy:", acc_gb)
print("Ensemble Accuracy:", acc_ensemble)

classifiers = ['SVM', 'Random Forest', 'Multinomial Naive Bayes', 'Logistic Regression', 'Gradient Boosting', 'Ensemble']
accuracies = [acc_svm, acc_rf, acc_nb, acc_lr, acc_gb, acc_ensemble]
plt.figure(figsize=(10, 6))
plt.plot(classifiers, accuracies, marker='o', linestyle='-', color='b')
plt.title('Classifier Comparison')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()