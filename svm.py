import feather
import pandas as pd
from sklearn import svm

#data intake and prep
train = feather.read_dataframe("C:\\Users\\riemang\\Documents\\MachineLearningPres\\Untitled\\trainset.feather")
lbls = train['y']
imgs = train.drop('y', 1)
test = feather.read_dataframe("C:\\Users\\riemang\\Documents\\MachineLearningPres\\Untitled\\testset.feather")
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