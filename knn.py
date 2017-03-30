import feather
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

train = feather.read_dataframe("C:\\Users\\riemang\\Documents\\MachineLearningPres\\Untitled\\trainset.feather")
lbls = train['y']
imgs = train.drop('y', 1)

knn = KNeighborsClassifier(n_neighbors=5)
# Fit the model on the training data.
knn.fit(imgs, lbls)

test = feather.read_dataframe("C:\\Users\\riemang\\Documents\\MachineLearningPres\\Untitled\\testset.feather")
lbls_test = test['y']
imgs_test = test.drop('y', 1)

# Make point predictions on the test set using the fit model.
predictions = knn.predict(imgs_test)
df_confusion = pd.crosstab(lbls_test, predictions, margins=True)
print(df_confusion)