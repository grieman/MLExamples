from mnist import MNIST
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

mndata = MNIST('C:\\Users\\riemang\\Documents\\Blog\\MLExamples\\mnist')
training = mndata.load_training()
imgs = pd.DataFrame(training[0])
labels = pd.Series(training[1])

knn = KNeighborsClassifier(n_neighbors=5)
# Fit the model on the training data.
knn.fit(imgs, labels)

# Make point predictions on the test set using the fit model.
testing = mndata.load_testing()
imgs = pd.DataFrame(testing[0])
labels = pd.Series(testing[1])
predictions = knn.predict(imgs)
df_confusion = pd.crosstab(labels, predictions, margins=True)
print(df_confusion)