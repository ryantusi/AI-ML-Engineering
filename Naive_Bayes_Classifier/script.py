"""
Email Similarity

In this project, you will use scikit-learn’s Naive Bayes implementation on several different datasets. By reporting the accuracy of the classifier, we can find which datasets are harder to distinguish. For example, how difficult do you think it is to distinguish the difference between emails about hockey and emails about soccer? How hard is it to tell the difference between emails about hockey and emails about tech? In this project, we’ll find out exactly how difficult those two tasks are.
"""

# 🧠 Email Similarity Project using Naive Bayes
# ---------------------------------------------
# By Ryan Tusi

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 1. Explore the dataset
emails = fetch_20newsgroups()
print("Target categories:\n", emails.target_names)

# 2. Select categories - baseball vs hockey
emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'])

# 3. Print one email
print("\nSample email at index 5:\n")
print(emails.data[5])

# 4. Print its label
print("\nLabel of email 5:", emails.target[5])
print("This corresponds to:", emails.target_names[emails.target[5]])

# 5. Create training set
train_emails = fetch_20newsgroups(
    categories=['rec.sport.baseball', 'rec.sport.hockey'],
    subset='train',
    shuffle=True,
    random_state=108
)

# 6. Create test set
test_emails = fetch_20newsgroups(
    categories=['rec.sport.baseball', 'rec.sport.hockey'],
    subset='test',
    shuffle=True,
    random_state=108
)

# 7. Create CountVectorizer
counter = CountVectorizer()

# 8. Fit to all data
counter.fit(test_emails.data + train_emails.data)

# 9. Transform training data
train_counts = counter.transform(train_emails.data)

# 10. Transform test data
test_counts = counter.transform(test_emails.data)

# 11. Create a Naive Bayes classifier
classifier = MultinomialNB()

# 12. Train the model
classifier.fit(train_counts, train_emails.target)

# 13. Test and print accuracy
print("\n🎯 Accuracy on Baseball vs Hockey dataset:")
print(classifier.score(test_counts, test_emails.target))

# 14. Try another dataset - Tech vs Hockey
train_emails = fetch_20newsgroups(
    categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'],
    subset='train',
    shuffle=True,
    random_state=108
)
test_emails = fetch_20newsgroups(
    categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'],
    subset='test',
    shuffle=True,
    random_state=108
)

# Fit counter again for the new categories
counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)

print("\n💻 Accuracy on Tech vs Hockey dataset:")
print(classifier.score(test_counts, test_emails.target))

# 15. Play with multiple categories (optional demo)
categories = ['rec.sport.hockey', 'sci.space', 'talk.politics.misc']
train_emails = fetch_20newsgroups(categories=categories, subset='train', shuffle=True, random_state=108)
test_emails = fetch_20newsgroups(categories=categories, subset='test', shuffle=True, random_state=108)

counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

classifier.fit(train_counts, train_emails.target)
print("\n🌍 Accuracy on Multiple Categories (Hockey, Space, Politics):")
print(classifier.score(test_counts, test_emails.target))
