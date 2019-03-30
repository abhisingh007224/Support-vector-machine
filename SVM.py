#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
                        
#Load the dataset

 dataset=pd.read_csv('# your csv file ')
 X= dataset.iloc[].values         # X is a independent variable
 y= dataset.iloc[].values         # y is a dependent variable

#split the dataset into training and test set
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,

                        random_state=0)

                      
#Apply feature scaling
  from sklearn.preprocessing import StandardScaler 

  sc=StandardScaler() 

  X_train=sc.fit_transform(X_train) 

  X_test=sc.transform(X_test) 

# create your classifier here
  from sklearn.svm import SVC 

  classifier=SVC(kernel='linear',random_state=0) 

  classifier.fit(X_train,y_train)

  y_pred=classifier.predict(X_test)

  from sklearn.metrics import confusion_matrix 

  cm=confusion_matrix(y_test,y_pred) 

#Visualising the Training set results
  from matplotlib.colors import ListedColormap 

  X_set, y_set = X_train, y_train

  X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                            stop = X_set[:, 0].max() + 1, step = 0.01),
                            np.arange(start = X_set[:, 1].min() - 1,
                            stop = X_set[:, 1].max() + 1, step = 0.01))

  plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
               X2.ravel()]).T).reshape(X1.shape),alpha = 0.75,
              cmap = ListedColormap(('red', 'green')))

  plt.xlim(X1.min(), X1.max())

  plt.ylim(X2.min(), X2.max())

  for i, j in enumerate(np.unique(y_set)):

            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                        c = ListedColormap(('red', 'green'))(i), label = j)

  plt.title('Kernel_SVM (Training set)')

  plt.xlabel('')

  plt.ylabel('')

  plt.legend()

  plt.show()
