# AdaBoost

自适应增强提升算法
一般是二分类算法

>AdaBoost全称为AdaptiveBoosting:自适应提升算法；虽然名字听起来给人一种高大上的感觉，但其实背后的原理并不难理解。什么叫做自适应，就是这个算法可以在不同的数据集上都适用,这个基本和废话一样,一个算法肯定要能适应不同的数据集。

提升方法是指:分类问题中，通过改变训练样本的权重，学习多个分类器，并将这些分类器进行线性组合，提高分类器的性能。

>但是标准的adaboost只适用于二分类任务。

## boosting过程

Boosting分类方法，其过程如下所示：

1.先通过对N个训练数据的学习得到第一个弱分类器h1；

2.将h1分错的数据和其他的新数据一起构成一个新的有N个训练数据的样本，通过对这个样本的学习得到第二个弱分类器h2；

3.将h1和h2都分错了的数据加上其他的新数据构成另一个新的有N个训练数据的样本，通过对这个样本的学习得到第三个弱分类器h3；

4.最终经过提升的强分类器h_final=Majority Vote(h1,h2,h3)。即某个数据被分为哪一类要通过h1,h2,h3的**多数表决**。
上述Boosting算法，存在两个问题：

**如何调整训练集，使得在训练集上训练弱分类器得以进行**。
**如何将训练得到的各个弱分类器联合起来形成强分类器**。

针对以上两个问题，AdaBoost算法进行了调整：

1.使用加权后选取的训练数据代替随机选取的训练数据，这样将**训练的焦点集中在比较难分的训练数据上**。

2.将弱分类器联合起来时，使用**加权的投票机制代替平均投票机制**。让**分类效果好的弱分类器具有较大的权重**，而**分类效果差的分类器具有较小的权重**。

这个很好理解:smile:

## 推导
[【参考知乎：一文弄懂AdaBoost】](https://zhuanlan.zhihu.com/p/59751960)

**训练数据**：$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots\left(x_{n}, y_{n}\right)\right\}$，其中$x_{i} \in \chi \subseteq R^{n}$，$y \in Y\{-1,+1\}$，最后需要得到分类器：$G(x)=\sum_{m=1}^{M} a_{m} G_{m}(x)$，其中 $m $为分类器的个数，每一次训练我们都获得一个基分类器 $G_{i}(x)$,$a_i$ 是每个基训练器的权重，也就是说每个基分类器说话的分量。我们看最后的分类器，他就是结合多个不同基分类器的意见，集百家之长，最终输出结果。

## 权重
那么每个基分类器的权重就显得十分重要了，那么这个权重是如何确定的呢，AdaBoost是这么考虑的，如果一个基分类器的准确率高，那么它的权重就会更高一点，反之权重就会较低

通常我们认为AdaBoost算法是模型为加法模型、损失函数为指数函数、学习算法为前向分步算法的二类分类学习方法。现在有出现了三个新名词


## 加法模型
三个关键字：基分类器，累加，权重


也就是一个函数（模型）是由**多个函数（模型）累加**起来的$f(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right)$

其中 $\beta_{m}$是每个基函数的系数， $\gamma_{m}$ 是每个基函数的参数， $b\left(x ; \gamma_{m}\right)$ 就是一个基函数了

假设一个基函数为 $e^{ax}$ ，那么一个加法模型就可以写成: $f(x)=e^{x}+2 e^{2 x}-2 e^{x / 2}$

## 前向分步算法

在给定训练数据以及损失函数 $L(y,f(x))$ 的情况下，加法模型的经验风险最小化即损失函数极小化问题如下:
$\min _{\beta_{m}, \gamma_{m}} \sum_{i=1}^{N} L\left(y_{i}, \sum_{m=1}^{N} \beta_{m} b\left(x ; \gamma_{m}\right)\right)$

这个问题直接优化比较困难，前向分步算法解决这个问题的思想如下:由于我们最终的分类器其实加法模型，所以我们可以从前向后考虑，**每增加一个基分类器，就使损失函数$L(y,f(x))$的值更小一点，逐步的逼近最优解**。这样考虑的话，**每一次计算损失函数的时候，我们只需要考虑当前基分类器的系数和参数**，同时**此次之前基分类器的系数和参数不受此次的影响**。算法的思想有点类似梯度下降，**每一次都向最优解移动一点**

步骤

**输入训练数据**：$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots\left(x_{n}, y_{n}\right)\right\}$，其中$x_{i} \in \chi \subseteq R^{n}$，$y \in Y\{-1,+1\}$，最后需要得到分类器$G(X)$

**初始化训练值的权值分布**
$D_{1}=\left(w_{1 i}, w_{2 i}, \ldots, w_{1 N}\right) ，
w_{1 i}=\frac{1}{N}$

对于$m=1,2,...,M$
**a**使用具有权值分布 $D_{m} 的训练数据集学习，得到基本分类器$G_{m}(x)$ 。

**b**计算 $G_{m}(x)$在训练集上的分类误差率

$e_{m}=\sum_{i=1}^{N} w_{m i} I\left\{y_{i} \neq G_{m}\left(x_{i}\right)\right\}$

**c**计算 $G_{m}(x)$的系数
$\alpha_{m}=\frac{1}{2} \log \frac{1-e_{m}}{e_{m}}$

**d**根据前m-1 次得到的结果，更新权值:
$w_{m=1, i}=\frac{w_{m i} e^{-y_{i} \alpha_{m} G_{m}\left(x_{i}\right)}}{Z_{m}}$

其中 $Z_{m}=\sum_{i=1}^{N} w_{m i} e^{-y_{i} \alpha_{m} G_{m}\left(x_{i}\right)}$,是一个规范化因子，用于归一化

**构建最终的分类器**
$f(x)=\sum_{m=1}^{M} a_{m} G_{m}(x)$
$G(x)=\operatorname{sign}(f(x))$

## boosting与adaboost的关系

提升树和AdaBoost之间的关系就好像编程语言中对象和类的关系，一个类可以生成多个不同的对象。提升树就是AdaBoost算法中基分类器选取决策树桩得到的算法。

用于分类的决策树主要有利用ID3和C4.5两种算法，我们选取任意一种算法，生成只有一层的决策树，即为决策树桩。

## 残差树

我们可以看到AdaBoost和提升树都是针对分类问题，如果是回归问题，上面的方法就不奏效了；而残差树则是针对回归问题的一种提升方法。其基学习器是基于CART算法的回归树，模型依旧为加法模型、损失函数为平方函数、学习算法为前向分步算法。

## 复现

参数
**base_estimator：**基分类器，默认是决策树，在该分类器基础上进行boosting，理论上可以是任意一个分类器，但是如果是其他分类器时需要指明样本权重

**n_estimators:**基分类器提升（循环）次数，默认是50次，这个值过大，模型容易过拟合；值过小，模型容易欠拟合。

**learning_rate:**学习率，表示梯度收敛速度，默认为1，如果过大，容易错过最优值，如果过小，则收敛速度会很慢；该值需要和n_estimators进行一个权衡，当分类器迭代次数较少时，学习率可以小一些，当迭代次数较多时，学习率可以适当放大。

**algorithm:boosting**算法，也就是模型提升准则，有两种方式SAMME, 和SAMME.R两种，默认是SAMME.R，两者的区别主要是弱学习器权重的度量，前者是对样本集预测错误的概率进行划分的，后者是对样本集的预测错误的比例，即错分率进行划分的，默认是用的SAMME.R。

**random_state:**随机种子设置。

### 属性
**estimators_:**以列表的形式返回所有的分类器。

**classes_:**类别标签

**estimator_weights_:**每个分类器权重

**estimator_errors_:**每个分类器的错分率，与分类器权重相对应。

**feature_importances_:**特征重要性，这个参数使用前提是基分类器也支持这个属性。

>关于Adaboost模型本身的参数并不多，但是我们在实际中除了调整Adaboost模型参数外，还可以调整基分类器的参数，关于基分类的调参，和单模型的调参是完全一样的，比如默认的基分类器是决策树，那么这个分类器的调参和我们之前的Sklearn参数详解——决策树是完全一致。

方法
decision_function(X):返回决策函数值（比如svm中的决策距离）

fit(X,Y):在数据集（X,Y）上训练模型。

get_parms():获取模型参数

**predict(X):**预测数据集X的结果。

predict_log_proba(X):预测数据集X的对数概率。

**predict_proba(X)**:预测数据集X的概率值。

score(X,Y):输出数据集（X,Y）在模型上的准确率。

staged_decision_function(X):返回每个基分类器的决策函数值

staged_predict(X):返回每个基分类器的预测数据集X的结果。

staged_predict_proba(X):返回每个基分类器的预测数据集X的概率结果。

staged_score(X, Y):返回每个基分类器的预测准确率。

datacamp的栗子

```python
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:,1]

# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))

<script.py> output:
    ROC AUC score: 0.71
```
