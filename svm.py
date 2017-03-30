import feather
import pandas as pd
from sklearn import svm

train = feather.read_dataframe("C:\\Users\\riemang\\Documents\\MachineLearningPres\\Untitled\\trainset.feather")
lbls = train['y']
imgs = train.drop('y', 1)

svm = svm.SVC()

svm.fit(imgs, lbls)
svm.score(imgs, lbls)

test = feather.read_dataframe("C:\\Users\\riemang\\Documents\\MachineLearningPres\\Untitled\\testset.feather")
lbls_test = test['y']
imgs_test = test.drop('y', 1)

predictions = svm.predict(imgs_test)
df_confusion = pd.crosstab(lbls_test, predictions, margins=True)
print(df_confusion)