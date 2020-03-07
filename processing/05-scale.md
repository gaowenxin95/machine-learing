# 标准化data


  特征数字差值很大的属性会对计算结果产生很大的影响，当我们认为特征是等权重的时候，因为取值范围不同，因此要进行归一化

|time    |distance|weight  |
| ------ | ------ | ------ |
|1.2|5000|80|
|1.6|6000|90|
|1.0|3000|50|
例如我们认为，time，distance，weight三个权重是一样的，在做特征分析的时候会明显发现distance对计算结果的影响是最大的。
因此，使用归一化的方法将数值处理到0~1的范围内

## 最值标准化方法

$x_{new}$=($x$-$x_{min}$)/($x_{max}$-$x_{min}$)
```r
cle<-function(df){
    df_new<-(df-min(df))/(max(df)-min(df))
    return df_new
}
```
## 均值方差标准化方法

$x_{\text {scale}}=\frac{x-x_{\text {mean}}}{s}$
```r
cle<-function(df){
    df_new<-(df-mean(df))/std(df)
    return df_new
}
```

python中提供了standardscaler类可以直接对np对象进行均值方差标准化
[可以参考](https://www.cnblogs.com/xuezou/p/9332763.html)

## scale
sklearn中常见标准化函数是StandardScaler

```python
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

<script.py> output:
    Accuracy with Scaling: 0.7700680272108843
    Accuracy without Scaling: 0.6979591836734694
```
很明显，标准化之后的数据的预测精度更高

## log归一化化数据

例如在回归模型中，因为在样本中的某些特征方差非常大，导致其他特征不起作用，因此需要训练模型之前先标准化数据。也就是缩放数据进行功能比较。

```r
In [1]: wine.head()
Out[1]: 
   Type  Alcohol   ...     OD280/OD315 of diluted wines  Proline
0     1    14.23   ...                             3.92     1065
1     1    13.20   ...                             3.40     1050
2     1    13.16   ...                             3.17     1185
3     1    14.37   ...                             3.45     1480
4     1    13.24   ...                             2.93      735

[5 rows x 14 columns]
```
比如Proline这个变量的存在就会导致其他变量不起作用，因此需要进行归一化。
常见方法可以使用log
```python
# Print out the variance of the Proline column
print(wine["Proline"].var())

# Apply the log normalization function to the Proline column
wine["Proline_log"] = np.log(wine["Proline"])

# Check the variance of the normalized Proline column
print(wine["Proline_log"].var())
<script.py> output:
    99166.71735542436
    0.17231366191842012
```


