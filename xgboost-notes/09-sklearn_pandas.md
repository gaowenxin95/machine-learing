# sklearn-pandas

>对 dataframe做特征工程时，简单的使用pandas自带的是可以实现的，但是稍微复杂一些的，我们就要使用sklearn 了，但是sklearn 特征工程没有办法直接操作 dataframe ，要么把dataframe 转化为 numpy 的array数组，但是会丢失索引和 列名，也担心再次组装时 会出现问题，有没有 可以让sklearn 直接操作 dataframe的可能？
有，当然有 ，那就是 sklearn-pandas ！！[简书]https://www.jianshu.com/p/29af03788ff6

## DataFrameMapper

针对列的转化器

>其声明中要配置好，针对哪些列做怎么样的特征转换操作

栗子[stackoverflow](https://stackoverflow.com/questions/30010853/using-onehotencoder-with-sklearn-pandas-dataframemapper)

[datacamp](https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/using-xgboost-in-pipelines?ex=9)

```python
# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                            [([numeric_feature],Imputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df=True,
                                            df_out=True
                                           )

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
                                                [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                input_df=True,
                                                df_out=True
                                               )
<script.py> output:
    age        9
    bp        12
    sg        47
    al        46
    su        49
    bgr       44
    bu        19
    sc        17
    sod       87
    pot       88
    hemo      52
    pcv       71
    wc       106
    rc       131
    rbc      152
    pc        65
    pcc        4
    ba         4
    htn        2
    dm         2
    cad        2
    appet      1
    pe         1
    ane        1
    dtype: int64


```


