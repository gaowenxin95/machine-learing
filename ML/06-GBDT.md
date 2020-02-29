# Gradient Boost Decision Tree
梯度增强决策树

## 定义

哈哈，偷个懒，mathpix这个月的免费次数用完了。。

```r
knitr::include_graphics("./figs/06.png")
```

<img src="./figs/06.png" width="746" />


[图片来源知乎](https://zhuanlan.zhihu.com/p/29765582)
GBDT的含义就是用Gradient Boosting的策略训练出来的DT模型。模型的结果是一组回归分类树组合(CART Tree Ensemble)：$T_1...T_K$ 。其中 $T_j$ 学习的是之前 $j-1$棵树预测结果的残差，这种思想就像准备考试前的复习，先做一遍习题册，然后把做错的题目挑出来，在做一次，然后把做错的题目挑出来在做一次，经过反复多轮训练，取得最好的成绩。[知乎](https://zhuanlan.zhihu.com/p/30339807)

目前我的理解就是：先随机抽取一些样本进行训练，得到一个基分类器，然后再次训练拟合模型的残差。
残差的定义：$y_{真实}-y_{预测}$，前一个基分类器未能拟合的部分也就是残差，于是新分类器继续拟合，直到残差达到指定的阈值。

## 基于残差的gradient

gradient是梯度的意思，也可以说是一阶导数

**平方损失函数MSE：**$\frac{1}{2} \sum_{0}^{n}\left(y_{i}-F\left(x_{i}\right)\right)^{2}$
熟悉其他算法的原理应该知道，这个损失函数主要针对回归类型的问题，分类则是用熵值类的损失函数。具体到平方损失函数的式子，你可能已经发现它的一阶导其实就是残差的形式，所以基于残差的GBDT是一种特殊的GBDT模型，它的损失函数是平方损失函数，常用来处理回归类的问题。具体形式可以如下表示：
**损失函数：**$L(y, F(x))=\frac{1}{2}(y-F(X))^{2}$
因此求最小化的$J=\frac{1}{2}(y-F(X))^{2}$
哈哈此使可以求一阶导数了
**损失函数的一阶导数（梯度）：**$\frac{\partial J}{\partial F\left(x_{i}\right)}=\frac{\partial \sum_{i} L\left(y_{i}, F\left(x_{i}\right)\right)}{\partial F\left(x_{i}\right)}=\frac{\partial L\left(y_{i}, F\left(x_{i}\right)\right)}{\partial F\left(x_{i}\right)}=F\left(x_{i}\right)-y_{i}$
而参数就是负的梯度：$y_{i}-F\left(x_{i}\right)=-\frac{\partial J}{\partial F\left(x_{i}\right)}$

### 评价
基于残差的GBDT在解决回归问题上不算是一个好的选择，一个比较明显的缺点就是对异常值过于敏感。
当存在一个异常值的时候，就会导致残差灰常之大。。自行理解

## boosting


gbdt模型可以认为是是由k个基模型组成的一个加法运算式

$\hat{y}_{i}=\sum_{k=1}^{K} f_{k}\left(x_{i}\right), f_{k} \in F$

其中F是指所有基模型组成的函数空间
那么一般化的损失函数是预测值 $\hat{y}_{i}$ 与 真实值$y_{i}$ 之间的关系，如我们前面的平方损失函数，那么对于n个样本来说，则可以写成
$L=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}\right)$

更一般的，我们知道一个好的模型，在偏差和方差上有一个较好的平衡，而算法的损失函数正是代表了模型的偏差面，最小化损失函数，就相当于最小化模型的偏差，但同时我们也需要兼顾模型的方差，所以目标函数还包括抑制模型复杂度的正则项，因此目标函数可以写成
$O b j=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}\right)+\sum_{k=1}^{K} \Omega\left(f_{k}\right)$
其中 $\Omega$ 代表了基模型的复杂度，若基模型是树模型，则树的深度、叶子节点数等指标可以反应树的复杂程度。

### 贪心算法
对于Boosting来说，它采用的是前向优化算法，即从前往后，逐渐建立基模型来优化逼近目标函数，具体过程如下：

$\hat{y}_{i}^{0}=0$
$\hat{y}_{i}^{1}=f_{1}\left(x_{i}\right)=\hat{y}_{i}^{0}+f_{1}\left(x_{i}\right)$
$\hat{y}_{i}^{2}=f_{1}\left(x_{i}\right)+f_{2}\left(x_{i}\right)=\hat{y}_{i}^{1}+f_{2}\left(x_{i}\right)$
$\cdots$
$\hat{y}_{i}^{t}=\sum_{k=1}^{t} f_{k}\left(x_{i}\right)=\hat{y}_{i}^{t-1}+f_{t}\left(x_{i}\right)$

### 如何学习一个新模型
关键还是在于GBDT的目标函数上，即新模型的加入总是以优化目标函数为目的的。

以第t步的模型拟合为例，在这一步，模型对第 $i$个样本 $x_i$ 的预测为：
$\hat{y}_{i}^{t}=\hat{y}_{i}^{t-1}+f_{t}\left(x_{i}\right)$

其中 $f_{t}\left(x_{i}\right)$ 就是我们这次需要加入的新模型，即需要拟合的模型，此时，目标函数就可以写成：

$\begin{aligned} O b j^{(t)} &=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{t}\right)+\sum_{i=i}^{t} \Omega\left(f_{i}\right) \\ &=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{t-1}+f_{t}\left(x_{i}\right)\right)+\Omega\left(f_{t}\right)+\text { constant } \end{aligned}$    (1)
因此当求出最优目标函数的时候也就相当于求出了$f_{t}\left(x_{i}\right)$ 


所以我么只要求出每一步损失函数的一阶和二阶导的值（由于前一步的 $\hat{y}_{i}^{t-1}$ 是已知的，所以这两个值就是常数）代入等式4，然后最优化目标函数，就可以得到每一步的 $f(x)$ ，最后根据加法模型得到一个整体模型

## demo

[看一个官方案例](https://mybinder.org/v2/gh/scikit-learn/scikit-learn/b194674c42d54b26137a456c510c5fdba1ba23e0?urlpath=lab%2Ftree%2Fnotebooks%2Fauto_examples%2Fensemble%2Fplot_gradient_boosting_regression.ipynb)



