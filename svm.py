import feather
import pandas as pd
from sklearn import svm
import os

path = os.getcwd()
trainpath = os.path.join(path, "trainset.feather")
testpath = os.path.join(path, "testset.feather")

#data intake and prep
train = feather.read_dataframe(trainpath)
lbls = train['y']
imgs = train.drop('y', 1)
test = feather.read_dataframe(testpath)
lbls_test = test['y']
imgs_test = test.drop('y', 1)

#initialize and train model
svm = svm.SVC(kernel='poly')
svm.fit(imgs, lbls)
svm.score(imgs, lbls)

#predict
predictions = svm.predict(imgs_test)
df_confusion = pd.crosstab(lbls_test, predictions, margins=True)
print(df_confusion)