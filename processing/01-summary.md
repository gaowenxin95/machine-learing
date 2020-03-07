# summary

再ML中“属性”==“特征”

## tidy data
整齐数据

我们平时所使用的数据都是经过整理的整齐数据(TidyData)，然而实际上我们接收到的数据很多都是杂乱无章的，为了进行数据的预处理，我们需要先把数据转换为整齐的数据.

```python
pd.shape
```
查询混数据框的维度


一般可以通过seaborn绘制pairplot图，完后观察一下异常特征
对于不需要的特征可以采取的方式的是,删掉指定列

```python
pd.drop("特征名",axis=1)
```
## 特征选择

- Filter：过滤法，按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。
- Wrapper：包装法，根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。
- Embedded：嵌入法，先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。

特征选择是从原始特征数据集中选择出子集，是一种包含的关系，没有更改原始的特征空间。

因为特征选择可以达到降维的目的，因此可以提高模型的泛化能力降低过拟合

## 特征提取

特征提取主要是通过属性间的关系，如组合不同的属性得到新的属性，这样就改变了原来的特征空间,新的特征可以是许多原有特征的线性组合。
>extracted features can be quite hard to interpret

提取的特征可能很难解释，哈哈，是这样的吧，因为新特征是原有特征的线性组合

- PCA
- LDA
- SVD


## 维数灾难

特征过多，因此需要选择有意义的特征进入模型训练，因此选择降维

## filter

### 方差阈值特征选择

也叫filter，就是移除方差低的
方差阈值（VarianceThreshold）

>是特征选择的一个简单方法，去掉那些方差没有达到阈值的特征。默认情况下，删除零方差的特征，例如那些只有一个值的样本。
假设我们有一个有布尔特征的数据集，然后我们想去掉那些超过80%的样本都是0（或者1）的特征。布尔特征是伯努利随机变量，方差为 p(1-p)。[zhilaizhiwang](https://www.jianshu.com/p/b3056d10a20f)

也就是选择方差大于阈值的特征

可以通过boxplot观察方差大小
可以直接使用sklearn中的VarianceThreshold方法

```python
from sklearn.feature_selection import VarianceThreshold

# Create a VarianceThreshold feature selector
sel =VarianceThreshold(threshold=0.001)

# Fit the selector to normalized head_df
sel.fit(head_df / head_df.mean())

# Create a boolean mask
mask = sel.get_support()

# Apply the mask to create a reduced dataframe
reduced_df = head_df.loc[:,mask]

print("Dimensionality reduced from {} to {}.".format(head_df.shape[1], reduced_df.shape[1]))

<script.py> output:
    Dimensionality reduced from 6 to 4.
```

## 特征中的缺失值的处理

先补充一个计算缺失率的方法

```python
df.isna().sum()/len(school_df)<p
```
增加p是筛选缺失率小于xxx的特征

## Pairwise correlation

两两的相关系数

计算相关系数可以通过.corr()

heat_map()可是查看热力图，也就是两两相关系数的图

datacamp上面的一个小的栗子

```python
# Create the correlation matrix
corr = ansur_df.corr()

# Generate a mask for the upper triangle 
mask = np.triu(np.ones_like(corr, dtype=bool))

# Add the mask to the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()
```



```r
knitr::include_graphics("./figs/02.png")
```

<img src="./figs/02.png" width="292" />

np.triu()返回三角矩阵，确实，heatmap返回上三角或者下三角就行了

## 去掉相关性强的特征值
person相关系数法，filter的一种，去掉相关性强的

np.ones_like 返回一个用1填充的跟输入 形状和类型 一致的数组

```python
# Calculate the correlation matrix and take the absolute value
# 计算相关系数
corr_matrix = ansur_df.corr().abs()

# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

# List column names of highly correlated features (r > 0.95)
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]

# Drop the features in the to_drop list
reduced_df = ansur_df.drop(to_drop, axis=1)

print("The reduced_df dataframe has {} columns".format(reduced_df.shape[1]))

<script.py> output:
    The reduced_df dataframe has 88 columns
```

## RFE
wrapper

递归式特征消除

>递归特征消除的主要思想是反复的构建模型（如SVM或者回归模型）然后选出最好的（或者最差的）的特征（可以根据coef来选），把选出来的特征选择出来，然后在剩余的特征上重复这个过程，直到所有特征都遍历了。这个过程中特征被消除的次序就是特征的排序。因此，这是一种寻找最优特征子集的贪心算法.[csdn](https://blog.csdn.net/a2099948768/article/details/82454135)

```python
class sklearn.feature_selection.RFE(estimator, n_features_to_select=None, step=1, verbose=0)
```

参数

- estimator:学习器

- n_features_to_select:特征选择的个数

- step：int or float,可选(default=1)如果大于等于1，step对应于迭代过程中每次移除的属性的数量（integer）。如果是（0.0，1.0），就对应于每次移除的特征的比例，四舍五入

属性

- n_features_：The number of selected features.

- support_：array of shape [n_features]
The mask of selected features.选择特征特征的bool型，优秀特征是true，不优秀的特征是false？

- ranking_:array of shape [n_features]
The feature ranking, such that ranking_[i] corresponds to the ranking position of the i-th feature. Selected (i.e., estimated best) features are assigned rank 1.输出第i个特征的排名位置

- estimator_：object
The external estimator fit on the reduced dataset.其他能减少数据集的估计器

```python
 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
 
iris = load_iris()
#特征矩阵
# print(a)
print(iris.data)
 
#目标向量
# print(iris.target)
 
#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
print(RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target))
 
```

## 基于树的特征选择
embedding

>基于树的预测模型（见 sklearn.tree 模块，森林见 sklearn.ensemble 模块）能够用来计算特征的重要程度 

```python
# Perform a 75% training and 25% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fit the random forest model to the training data
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Calculate the test set accuracy
acc = accuracy_score(y_test, rf.predict(X_test))

# Print the importances per feature
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# Print accuracy
print("{0:.1%} accuracy on test set.".format(acc)) 

<script.py> output:
    {'diastolic': 0.08, 'pregnant': 0.09, 'age': 0.16, 'insulin': 0.13, 'glucose': 0.21, 'family': 0.12, 'bmi': 0.09, 'triceps': 0.11}
    77.6% accuracy on test set.
```

 - get_support 方法来查看哪些特征被选中，它会返回所选特征的布尔遮罩（mask）[参考](https://www.cnblogs.com/stevenlk/p/6543628.html)

## 正则化线性回归

- LassoCV 
- RidgeCV

```python 
from sklearn.linear_model import LassoCV

# Create and fit the LassoCV model on the training set
lcv = LassoCV()
lcv.fit(X_train, y_train)
print('Optimal alpha = {0:.3f}'.format(lcv.alpha_))

# Calculate R squared on the test set
r_squared = lcv.score(X_test, y_test)
print('The model explains {0:.1%} of the test set variance'.format(r_squared))

# Create a mask for coefficients not equal to zero
lcv_mask = lcv.coef_ != 0
print('{} features out of {} selected'.format(sum(lcv_mask), len(lcv_mask)))

Optimal alpha = 0.089
The model explains 88.2% of the test set variance
26 features out of 32 selected
```


## 集成器特征选择器

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor

# Select 10 features with RFE on a GradientBoostingRegressor, drop 3 features on each step
rfe_gb = RFE(estimator=GradientBoostingRegressor(), 
             n_features_to_select=10, step=3, verbose=1)
rfe_gb.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_gb.score(rfe_gb,y_test)
print('The model can explain {0:.1%} of the variance in the test set'.format(r_squared))

<script.py> output:
    Fitting estimator with 32 features.
    Fitting estimator with 29 features.
    Fitting estimator with 26 features.
    Fitting estimator with 23 features.
    Fitting estimator with 20 features.
    Fitting estimator with 17 features.
    Fitting estimator with 14 features.
    Fitting estimator with 11 features.
```

可以有一个集成器，也可以组合多个集成器，or学习器




