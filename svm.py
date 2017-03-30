from mnist import MNIST
import pandas as pd
import seaborn as sns
from sklearn import svm

mndata = MNIST('C:\\Users\\riemang\\Documents\\Blog\\MLExamples\\mnist')
training = mndata.load_training()
imgs = pd.DataFrame(training[0])
labels = pd.Series(training[1])

svm = svm.SVC()

svm.fit(imgs, labels)
svm.score(imgs, labels)

testing = mndata.load_testing()
imgs = pd.DataFrame(testing[0])
labels = pd.Series(testing[1])
predictions = svm.predict(imgs)
df_confusion = pd.crosstab(labels, predictions, margins=True)
print(df_confusion)