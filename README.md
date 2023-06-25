#基础函数库

将numpy导入为np

将熊猫作为pd导入

#导入画图库

将matplotlib.pylot导入为plt

将Seaborn导入为sns

## 导入逻辑回归模型函数

从sklearn.linearn_model导入LogisticRegression

##演示演示logisticRegression分类



## 构造数据集

X_foreures=np.array([[-1, -2],[-2, -1],[-3, -2],[1, 3],[2, 1],[3, 2]])

y_label=np.array([0, 0, 0, 1, 1, 1])





## 调用逻辑回归模型

LR_clf=LogisticRegression()#实例化LogisticRegression()类

## 用逻辑回归模型拟合构造的数据集

LR_clf=lr_clf.fit(x_fearures，y_label)#其拟合方程为y=w0+w1*X1+w2*X2

打印(x_fearures[:,0])

打印(x_fearures[:,1])

##查看其对应模型的w

print('Logistic回归的权重：'，lr_clf.coef_)

##查看其对应模型的w0

print('logistic回归的截距(w0)：'，lr_clf.intercept_)

##Logistic回归的权重：[[0.73462087 0.6947908]]

##the intercept(w0) of Logistic Regression:[-0.03643213]

## 可视化构造的数据样本点

# 可视化决策边界

plt.figure()

plt.scatter(x_fearures[:,0],x_fearures[:,1], c=y_label, s=50, cmap='viridis')

plt.title('Dataset')

plt.show()

nx, ny = 200, 100

x_min, x_max = plt.xlim()

y_min, y_max = plt.ylim()

x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))



z_proba = lr_clf.predict_proba(np.c_[x_grid.ravel(), y_grid.ravel()])

z_proba = z_proba[:, 1].reshape(x_grid.shape)

plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')

plt.show()

### 可视化预测新样本



plt.figure()

## new point 1

x_fearures_new1 = np.array([[0, -1]])

plt.scatter(x_fearures_new1[:,0],x_fearures_new1[:,1], s=50, cmap='viridis')

plt.annotate(s='New point 1',xy=(0,-1),xytext=(-2,0),color='blue',arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))

new point 2

x_fearures_new2 = np.array([[1, 2]])

plt.scatter(x_fearures_new2[:,0],x_fearures_new2[:,1], s=50, cmap='viridis')

plt.annotate(s='New point 2',xy=(1,2),xytext=(-1.5,2.5),color='red',arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))

plt.scatter(x_fearures[:,0],x_fearures[:,1], c=y_label, s=50, cmap='viridis')

plt.title('Dataset')



# 可视化决策边界

plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')


##在训练集和测试集上分布利用训练好的模型进行预测

y_label_new1_predict=lr_clf.predict(x_fearures_new1)

y_label_new2_predict=lr_clf.predict(x_fearures_new2)

print('The New point 1 predict class:\n',y_label_new1_predict)

print('The New point 2 predict class:\n',y_label_new2_predict)

##由于逻辑回归模型是概率预测模型（前文介绍的p = p(y=1|x,\theta)）,所有我们可以利用predict_proba函数预测其概率

y_label_new1_predict_proba=lr_clf.predict_proba(x_fearures_new1)

y_label_new2_predict_proba=lr_clf.predict_proba(x_fearures_new2)

print('The New point 1 predict Probability of each class:\n',y_label_new1_predict_proba)

print('The New point 2 predict Probability of each class:\n',y_label_new2_predict_proba)

##我们利用sklearn中自带的iris数据作为数据载入，并利用Pandas转化为DataFrame格式

from sklearn.datasets import load_iris

data = load_iris() #得到数据特征

print(data['DESCR'])



iris_target = data.target #得到数据对应的标签

iris_features = pd.DataFrame(data=data.data, columns=data.feature_names) #利用Pandas转化为DataFrame格式

##利用.info()查看数据的整体信息

iris_features.info()

iris_features.head()

iris_features.tail()

iris_features.describe()

pd.Series(iris_target).value_counts()

## 合并标签和特征信息

iris_all = iris_features.copy() ##进行浅拷贝，防止对于原始数据的修改

iris_all['target'] = iris_target

## 特征与标签组合的散点可视化

sns.pairplot(data=iris_all,diag_kind='hist', hue= 'target')

plt.show()



for col in iris_features.columns:

    sns.boxplot(x='target', y=col, saturation=0.5,

palette='pastel', data=iris_all)

    plt.title(col)

    plt.show()

# 选取其前三个特征绘制三维散点图

from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')



iris_all_class0 = iris_all[iris_all['target']==0].values

iris_all_class1 = iris_all[iris_all['target']==1].values

iris_all_class2 = iris_all[iris_all['target']==2].values

# 'setosa'(0), 'versicolor'(1), 'virginica'(2)


ax.scatter(iris_all_class0[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='setosa')

ax.scatter(iris_all_class1[:,0], iris_all_class1[:,1], iris_all_class1[:,2],label='versicolor')

ax.scatter(iris_all_class2[:,0], iris_all_class2[:,1], iris_all_class2[:,2],label='virginica')

plt.legend()

plt.show()

iris_features.iloc[:100]



# 选取其前三个特征绘制三维散点图

from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')



iris_all_class0 = iris_all[iris_all['target']==0].values

iris_all_class0[:,0]

##为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。

from sklearn.model_selection import train_test_split

##选择其类别为0和1的样本（不包括类别为2的样本）

iris_features_part=iris_features.iloc[:100]

iris_target_part=iris_target[:100]

#前100样本为0，1类样本

#后50样本为2类样本

##测试集大小为20%，80%/20%分

x_train,x_test,y_train,y_test=train_test_split(iris_features_part,iris_target_part,test_size=0.2,random_state=2020)


##从sklearn中导入逻辑回归模型

from sklearn.linear_model import LogisticRegression

##定义逻辑回归模型

clf=LogisticRegression(random_state=0,solver='lbfgs')

##在训练集上训练逻辑回归模型

clf.fit(x_train,y_train)

##查看其对应的w

print('the weight of Logistic Regression:',clf.coef_)



##查看其对应的w0

print('the intercept(w0) of Logistic Regression:',clf.intercept_)

##在训练集和测试集上分布利用训练好的模型进行预测

train_predict=clf.predict(x_train)

test_predict=clf.predict(x_test)



from sklearn import metrics

##利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果

print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))

print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

##查看混淆矩阵(预测值和真实值的各类情况统计矩阵)

confusion_matrix_result=metrics.confusion_matrix(test_predict,y_test)

print('The confusion matrix result:\n',confusion_matrix_result)



##利用热力图对于结果进行可视化

plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix_result,annot=True,cmap='Blues')

plt.xlabel('Predictedlabels')

plt.ylabel('Truelabels')

plt.show()

#利用 逻辑回归模型 在三分类(多分类)上 进行训练和预测



##测试集大小为20%，80%/20%分

x_train,x_test,y_train,y_test=train_test_split(iris_features,iris_target,test_size=0.2,random_state=2020)

##定义逻辑回归模型

clf=LogisticRegression(random_state=0,solver='lbfgs')

##在训练集上训练逻辑回归模型

clf.fit(x_train,y_train)

##查看其对应的w

print('the weight of Logistic Regression:\n',clf.coef_)

##查看其对应的w0

print('the intercept(w0) of Logistic Regression:\n',clf.intercept_)

##由于这个是3分类，所有我们这里得到了三个逻辑回归模型的参数，其三个逻辑回归组合起来即可实现三分类

##在训练集和测试集上分布利用训练好的模型进行预测

train_predict=clf.predict(x_train)

test_predict=clf.predict(x_test)

##由于逻辑回归模型是概率预测模型（前文介绍的p=p(y=1|x,\theta)）,所有我们可以利用predict_proba函数预测其概率



train_predict_proba=clf.predict_proba(x_train)

test_predict_proba=clf.predict_proba(x_test)



print('The test predict Probability of each class:\n',test_predict_proba)

##其中第一列代表预测为0类的概率，第二列代表预测为1类的概率，第三列代表预测为2类的概率。

##利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果

print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))

print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))



##查看混淆矩阵

confusion_matrix_result=metrics.confusion_matrix(test_predict,y_test)

print('The confusion matrix result:\n',confusion_matrix_result)

##利用热力图对于结果进行可视化

plt.figure(figsize=(8,6))

sns.heatmap(confusion_matrix_result,annot=True,cmap='Blues')

plt.xlabel('Predicted labels')

plt.ylabel('True labels')

plt.show()
