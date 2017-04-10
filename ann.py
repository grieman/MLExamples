import feather, os
import pandas as pd
from sklearn.neural_network import MLPClassifier #requires 0.18

path = os.getcwd()
trainpath = os.path.join(path, "trainset.feather")
testpath = os.path.join(path, "testset.feather")

train = feather.read_dataframe(trainpath)
lbls = train['y']
imgs = train.drop('y', 1)
test = feather.read_dataframe(testpath)
lbls_test = test['y']
imgs_test = test.drop('y', 1)

#initialize and train model
mlp = MLPClassifier(hidden_layer_sizes=(10))
mlp.fit(imgs, lbls)

#predict
predictions = mlp.predict(imgs_test)
df_confusion = pd.crosstab(lbls_test, predictions, margins=True)
print(df_confusion)