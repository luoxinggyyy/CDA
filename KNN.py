import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
train_set=pd.read_table(r'D:\fruit1.txt')
label=np.array(train_set['fruit1_label'])
train= pd.concat([train_set['Capacitance'],train_set['Resistance'],train_set['qqq'],train_set['www']],axis=1)
train=np.array(train)
train
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
train=scaler.fit_transform(train)
laberl=np.array(train_set['fruit1_label'])
X_train,X_test,y_train,y_test=train_test_split(train,label,train_size=0.8,random_state=1)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print("model accuracy:",metrics.accuracy_score(y_test,y_pred))
    X_mat = X[['Capacitance', 'Resistance']].values
    y_mat = y.values
 
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF','#FFC0CB','#800080'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF','#FFC0CB','#800080'])
 
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(X_mat, y_mat)
 
    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.
    
    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50
    
    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    d=np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(d)
 
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 
    # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold,
                edgecolor='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
 
    patch0 = mpatches.Patch(color='#FF0000', label='100HZ_in_97%RH')
    patch1 = mpatches.Patch(color='#00FF00', label='100HZ_in_33%RH')
    patch2 = mpatches.Patch(color='#0000FF', label='100HZ_in_75%RH')
    patch3 = mpatches.Patch(color='#AFAFAF', label='1kHZ_in_33%RH')
    patch4 = mpatches.Patch(color='#FFC0CB', label='1kHZ_in_75%RH')
    patch5 = mpatches.Patch(color='#800080', label='1kHZ_in_97%RH')
    plt.legend(handles=[patch0, patch1, patch2, patch3, patch4, patch5])
 
    plt.xlabel('Capacitance (cm)')
    plt.ylabel('Resistance (cm)')
    plt.show()