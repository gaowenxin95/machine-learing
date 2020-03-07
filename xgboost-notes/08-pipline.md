# pipline

使用管道函数
Have fun building pipline

就相当于magrittr里面的 %>% ，可以完成多个步骤先后执行

>Pipeline可以用于把多个estimators级联合成一个estimator。这么做的原因是考虑了数据处理过程的一系列前后相继的固定流程

pipline可以完成两件事情：

- 只需要一次fit和predict就可以在训练集上训练一组estimator
- 可以结合grid进行调参


```python
# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict("records"), y, cv=10, scoring="neg_mean_squared_error")

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))

<script.py> output:
    [12:43:18] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [12:43:19] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [12:43:19] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [12:43:20] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [12:43:20] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [12:43:21] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [12:43:21] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [12:43:22] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [12:43:22] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [12:43:23] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    10-fold RMSE:  29867.603720688923
```


>机器学习中的类别特征包括离散型和连续型

## 连续特征标准化

拿到获取的原始特征，必须对每一特征分别进行**标准化**，比如，特征A的取值范围是[-1000,1000]，特征B的取值范围是[-1,1].如果使用logistic回归，w1*x1+w2*x2，因为x1的取值太大了，所以x2基本起不了作用。所以，必须进行特征的归一化，每个特征都单独进行归一化。[引自知乎](https://zhuanlan.zhihu.com/p/35287916)

但是基于树的模型不需要标准化

标准化包括：

最值标准化
均值标准化[cnblog](https://www.cnblogs.com/gaowenxingxing/p/12295207.html)

## 离散特征标准化

Binarize categorical/discrete features: 对于离散的特征基本就是按照one-hot（独热）编码，该离散特征有多少取值，就用多少维来表示该特征。

## OneHotEncoder

独热编码[引自知乎](https://zhuanlan.zhihu.com/p/35287916)


独热码，在英文文献中称做 one-hot code, 直观来说就是有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制。举例如下：

>假如有三种颜色特征：红、黄、蓝。
在利用机器学习的算法时一般需要进行向量化或者数字化。那么你可能想令 红=1，黄=2，蓝=3. 那么这样其实实现了标签编码，即给不同类别以标签。然而这意味着机器可能会学习到“红<黄<蓝”，但这并不是我们的让机器学习的本意，只是想让机器区分它们，并无大小比较之意。所以这时标签编码是不够的，需要进一步转换。因为有三种颜色状态，所以就有3个比特。即红色：1 0 0 ，黄色: 0 1 0，蓝色：0 0 1 。如此一来每两个向量之间的距离都是根号2，在向量空间距离都相等，所以这样不会出现偏序性，基本不会影响基于向量空间度量算法的效果。

自然状态码为：000,001,010,011,100,101

独热编码为：000010,000010,000100,001000,010000,100000

```python
# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categorical_features=categorical_mask, sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)

<script.py> output:
    [[0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00 0.000e+00 0.000e+00
      1.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 1.000e+00 0.000e+00 0.000e+00 6.000e+01 6.500e+01 8.450e+03
      7.000e+00 5.000e+00 2.003e+03 0.000e+00 1.710e+03 1.000e+00 0.000e+00
      2.000e+00 1.000e+00 3.000e+00 0.000e+00 5.480e+02 2.085e+05]
     [0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00 0.000e+00 0.000e+00
      1.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00 1.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 2.000e+01 8.000e+01 9.600e+03
      6.000e+00 8.000e+00 1.976e+03 0.000e+00 1.262e+03 0.000e+00 1.000e+00
      2.000e+00 0.000e+00 3.000e+00 1.000e+00 4.600e+02 1.815e+05]
     [0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00 0.000e+00 0.000e+00
      1.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 1.000e+00 0.000e+00 0.000e+00 6.000e+01 6.800e+01 1.125e+04
      7.000e+00 5.000e+00 2.001e+03 1.000e+00 1.786e+03 1.000e+00 0.000e+00
      2.000e+00 1.000e+00 3.000e+00 1.000e+00 6.080e+02 2.235e+05]
     [0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00 0.000e+00 0.000e+00
      1.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      1.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 1.000e+00 0.000e+00 0.000e+00 7.000e+01 6.000e+01 9.550e+03
      7.000e+00 5.000e+00 1.915e+03 1.000e+00 1.717e+03 1.000e+00 0.000e+00
      1.000e+00 0.000e+00 3.000e+00 1.000e+00 6.420e+02 1.400e+05]
     [0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00 0.000e+00 0.000e+00
      1.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 1.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00
      0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
      0.000e+00 1.000e+00 0.000e+00 0.000e+00 6.000e+01 8.400e+01 1.426e+04
      8.000e+00 5.000e+00 2.000e+03 0.000e+00 2.198e+03 1.000e+00 0.000e+00
      2.000e+00 1.000e+00 4.000e+00 1.000e+00 8.360e+02 2.500e+05]]
    (1460, 21)
    (1460, 62)
```


### 为什么要独热编码？

正如上文所言，独热编码（哑变量 dummy variable）是因为大部分算法是基于向量空间中的度量来进行计算的，为了使非偏序关系的变量取值不具有偏序性，并且到圆点是等距的。使用one-hot编码，将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点。将离散型特征使用one-hot编码，会让特征之间的距离计算更加合理。离散特征进行one-hot编码后，编码后的特征，其实每一维度的特征都可以看做是连续的特征。就可以跟对连续型特征的归一化方法一样，对每一维特征进行归一化。比如归一化到[-1,1]或归一化到均值为0,方差为1。

### 为什么特征向量要映射到欧式空间？

将离散特征通过one-hot编码映射到欧式空间，是因为，在回归，分类，聚类等机器学习算法中，特征之间距离的计算或相似度的计算是非常重要的，而我们常用的距离或相似度的计算都是在欧式空间的相似度计算，计算余弦相似性，基于的就是欧式空间。


### 独热编码优缺点

**优点：**独热编码解决了分类器不好处理属性数据的问题，在一定程度上也起到了扩充特征的作用。它的值只有0和1，不同的类型存储在垂直的空间。
缺点：当类别的数量很多时，特征空间会变得非常大。在这种情况下，一般可以用PCA来减少维度。而且**one hot encoding+PCA**这种组合在实际中也非常有用。


### 什么情况下(不)用独热编码？

- 用：独热编码用来解决类别型数据的离散值问题，

- 不用：将离散型特征进行one-hot编码的作用，是为了让距离计算更合理，但如果特征是离散的，并且不用one-hot编码就可以很合理的计算出距离，那么就没必要进行one-hot编码。 有些基于树的算法在处理变量时，并不是基于向量空间度量，数值只是个类别符号，即没有偏序关系，所以不用进行独热编码。 

- Tree Model不太需要one-hot编码： 对于决策树来说，one-hot的本质是增加树的深度。
总的来说，要是one hot encoding的类别数目不太多，建议优先考虑。



###  什么情况下(不)需要归一化？

需要： 基于参数的模型或基于距离的模型，都是要进行特征的归一化。
不需要：基于树的方法是不需要进行特征的归一化，例如随机森林，bagging 和 boosting等。


## LabelEncoder

标签编码

计算机没办法识别文本特征，因此需要把文本进行转化

作用： 利用LabelEncoder()将转换成连续的数值型变量。即是对不连续的数字或者文本进行编号例如：

```python
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
# 要经过fit_transform
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())

<script.py> output:
      MSZoning PavedDrive Neighborhood BldgType HouseStyle
    0       RL          Y      CollgCr     1Fam     2Story
    1       RL          Y      Veenker     1Fam     1Story
    2       RL          Y      CollgCr     1Fam     2Story
    3       RL          Y      Crawfor     1Fam     2Story
    4       RL          Y      NoRidge     1Fam     2Story
       MSZoning  PavedDrive  Neighborhood  BldgType  HouseStyle
    0         3           2             5         0           5
    1         3           2            24         0           2
    2         3           2             5         0           5
    3         3           2             6         0           5
    4         3           2            15         0           5
```

## DictVectorizer

文本特征向量化的方法？
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html

这个貌似是文本分析里面的，我先放个栗子，后面补充


## embedding


第二个方法处理离散特征的方法[机器学习与自然语言处理](https://zhuanlan.zhihu.com/p/40094324)
是特征嵌入embedding。这个一般用于深度学习中。比如对于用户的ID这个特征，如果要使用独热编码，则**维度会爆炸**，如果使用特征嵌入就维度低很多了。对于每个要嵌入的特征，我们会有一个特征嵌入矩阵，这个矩阵的行很大，对应我们该特征的数目。比如用户ID，如果有100万个，那么嵌入的特征矩阵的行就是100万。但是列一般比较小，比如可以取20。这样每个用户ID就转化为了一个20维的特征向量。进而参与深度学习模型。在tensorflow中，我们可以先随机初始化一个特征嵌入矩阵，对于每个用户，可以用tf.nn.embedding_lookup找到该用户的特征嵌入向量。特征嵌入矩阵会在反向传播的迭代中优化。


**此外，在自然语言处理中，我们也可以用word2vec将词转化为词向量，进而可以进行一些连续值的后继处理**


## Feature Union

特征融合

>FeatureUnion把若干个transformer object组合成一个新的estimators。这个新的transformer组合了他们的输出，一个FeatureUnion对象接受一个transformer对象列表。

在训练阶段，每一个transformer都在数据集上独立的训练。在数据变换阶段，多有的训练好的Trandformer可以并行的执行。他们输出的样本特征向量被以end-to-end的方式拼接成为一个更大的特征向量。

在这里，FeatureUnion提供了两种服务：

- Convenience： 你只需要调用一次fit和transform就可以在数据集上训练一组estimators。

- Joint parameter selection： 可以把grid search用在FeatureUnion中所有的estimators的参数这上面。
FeatureUnion和Pipeline可以组合使用来创建更加复杂的模型。[作者：cnkai](https://www.jianshu.com/p/c532424541ad)

FeatureUnion对象实例使用（key，value）构成的list来构造，key是你自己起的transformation的名称，value是一个estimator对象。

datacamp上面的一个小例子

```python
# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)
                                         ])
                                         
# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(max_depth=3))
                    ])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, kidney_data, y, scoring="roc_auc", cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))

<script.py> output:
    3-fold AUC:  0.998637406769937
```



