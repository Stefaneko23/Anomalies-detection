import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score



class SVM:
    def __init__(self, learning_rate=0.01, epochs=200, C=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Initialize model parameters
        self.w = np.zeros(X.shape[1])
        self.b = 0

        # Gradient descent
        for _ in range(self.epochs):
            # Compute gradients
            gradient_w = self.w
            gradient_b = 0

            for i in range (len (y)):
              if(y[i]*(np.dot(X[i,:], self.w)+ self.b)<1):
                gradient_w -=self.C*(-y[i]) * X[i]
                gradient_b -=self.C*(-y[i])

            # Update weights and bias
            self.w -= self.learning_rate * gradient_w
            self.b -= self.learning_rate * gradient_b

    def predict(self, X):
        # Compute predictions
        final_predictions = []
        predictions = np.sign(X.dot(self.w)+self.b)
        for prediction in predictions:
            if prediction < 0:
                final_predictions.append(0.0)
            else:
                final_predictions.append(1.0)

        return final_predictions

testfile = open('test.csv')
trainfile = open('train.csv')

test_csvreader = csv.reader(testfile)
train_csvreader = csv.reader(trainfile)

train=[]
train_ids = []
train_labels = []

header=next(train_csvreader)

for row in train_csvreader:
    float_row = [round(float(item), 2) for item in row]
    train_ids.append(float_row[0])
    train.append(float_row[1:7])
    train_labels.append(float_row[7])

test=[] 
test_ids = []

header=next(test_csvreader)

for row in test_csvreader:
    float_row = [float(item) for item in row]
    test_ids.append(float_row[0])
    test.append(float_row[1:7])

# # Define the feature names
feature_names = ['incidence', 'tilt', 'angle', 'slope', 'radius', 'grade']

# Create a DataFrame from the array data
df = pd.DataFrame(train, columns=feature_names)

 # Create a pair plot
sns.pairplot(df)
plt.show()


train = np.array(train)
test = np.array(test)
#--------------------------
scaler = StandardScaler()
train = scaler.fit_transform(train)



svm = SVM()
svm.fit(train, train_labels)

test_labels =svm.predict(test)

#--------------------------

output_formatat = []
for i in range(len(test)):
    output_formatat.append(str(test_ids[i]) + ',' + str(test_labels[i]))


np.savetxt('submission.csv', output_formatat, "%s", header="id,is_anomaly", comments="")



# pred =svm.predict(train)
# precision = precision_score(train_labels, pred)
# recall = recall_score(train_labels, pred)
# f1 = f1_score(train_labels, pred)
# accuracy = accuracy_score(train_labels, pred)
# print(precision, recall, f1, accuracy)
