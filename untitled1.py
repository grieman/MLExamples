from mnist import MNIST
import pandas as pd
from sklearn.decomposition import RandomizedPCA
import seaborn as sns


mndata = MNIST('C:\\Users\\riemang\\Documents\\Blog\\MLExamples\\mnist')
training = mndata.load_training()
imgs = pd.DataFrame(training[0])
labels = pd.Series(training[1])


pca = RandomizedPCA(n_components=2)
X = pca.fit_transform(imgs)
X = pd.DataFrame(X)
X.columns = ['PC1', 'PC2']
X['class'] = training[1]


sns.set_palette(sns.color_palette("Set3", 10))
sns.lmplot('PC1', 'PC2', data=X, hue='class', fit_reg=False).savefig('C:\\Users\\riemang\\Documents\\Blog\\MLExamples\\Images\\python_PCA.png')
