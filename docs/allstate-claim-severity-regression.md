# Allstate claims severity

## Introduction

- Allstate is a personal insurer in United State, continually seeking fresh ideas to improve their claims service for the over 16 million households they protect.
- Allstate wish to  develop automated methods of predicting the cost, and hence severity, of claims.
- Particularly this problem deals with predicting the claims severity pertaining to car insurance. 

## Business problem

Definition of claim severity : It is the measure of loss to insurance company associated with an insurance claim made by a customer.
- An insurance company tries to minimize the average claim severity over all claims for the period in consideration.
- By the measure of claim severity insurance company gets an idea about the risk associated with the insurance.
- By fast and accurate prediction of claim severity insurance company would be able to make faster decision whether to provide insurance to a person based on the risk factor.
- Claim severity would also serve the purpose of deciding the cost of insurance for a customer as per his/her details.
- It will reduce the operation cost and time and also provide better customer experience.

## Machine Leaning problem

- Independent variables : Mixture of categorical and numerical features.
- Dependent variable: A numerical value (loss)

**This is a regression problem** <br><br>
The performance metric we will use here is mean absolute error (MAE) between actual loss and predict loss.

## Dataset overview

Some high level observations about the dataset given are :
- There are 116 categorical features named as cat1, cat2, cat3 ...... cat116
- There are 14 real valued features named as cont1, cont2, cont3...... cont14
- One target variable named as loss.

## Exploratory Data Analysis (EDA)

### Importing libraries and modules


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from category_encoders.binary import BinaryEncoder
from sklearn.metrics import mean_absolute_error
from os import path
import joblib
import os
from xgboost import XGBRegressor
from category_encoders.one_hot import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
%matplotlib inline
plt.style.use('seaborn')
import warnings
warnings.filterwarnings("ignore")

```

### Importing the dataset


```python
data_df = pd.read_csv('train.csv')
print('Shape: ',data_df.shape)
print('Columns: ',data_df.columns.values)
data_df.head()
```

    Shape:  (188318, 132)
    Columns:  ['id' 'cat1' 'cat2' 'cat3' 'cat4' 'cat5' 'cat6' 'cat7' 'cat8' 'cat9'
     'cat10' 'cat11' 'cat12' 'cat13' 'cat14' 'cat15' 'cat16' 'cat17' 'cat18'
     'cat19' 'cat20' 'cat21' 'cat22' 'cat23' 'cat24' 'cat25' 'cat26' 'cat27'
     'cat28' 'cat29' 'cat30' 'cat31' 'cat32' 'cat33' 'cat34' 'cat35' 'cat36'
     'cat37' 'cat38' 'cat39' 'cat40' 'cat41' 'cat42' 'cat43' 'cat44' 'cat45'
     'cat46' 'cat47' 'cat48' 'cat49' 'cat50' 'cat51' 'cat52' 'cat53' 'cat54'
     'cat55' 'cat56' 'cat57' 'cat58' 'cat59' 'cat60' 'cat61' 'cat62' 'cat63'
     'cat64' 'cat65' 'cat66' 'cat67' 'cat68' 'cat69' 'cat70' 'cat71' 'cat72'
     'cat73' 'cat74' 'cat75' 'cat76' 'cat77' 'cat78' 'cat79' 'cat80' 'cat81'
     'cat82' 'cat83' 'cat84' 'cat85' 'cat86' 'cat87' 'cat88' 'cat89' 'cat90'
     'cat91' 'cat92' 'cat93' 'cat94' 'cat95' 'cat96' 'cat97' 'cat98' 'cat99'
     'cat100' 'cat101' 'cat102' 'cat103' 'cat104' 'cat105' 'cat106' 'cat107'
     'cat108' 'cat109' 'cat110' 'cat111' 'cat112' 'cat113' 'cat114' 'cat115'
     'cat116' 'cont1' 'cont2' 'cont3' 'cont4' 'cont5' 'cont6' 'cont7' 'cont8'
     'cont9' 'cont10' 'cont11' 'cont12' 'cont13' 'cont14' 'loss']
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>...</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.718367</td>
      <td>0.335060</td>
      <td>0.30260</td>
      <td>0.67135</td>
      <td>0.83510</td>
      <td>0.569745</td>
      <td>0.594646</td>
      <td>0.822493</td>
      <td>0.714843</td>
      <td>2213.18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.438917</td>
      <td>0.436585</td>
      <td>0.60087</td>
      <td>0.35127</td>
      <td>0.43919</td>
      <td>0.338312</td>
      <td>0.366307</td>
      <td>0.611431</td>
      <td>0.304496</td>
      <td>1283.60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.289648</td>
      <td>0.315545</td>
      <td>0.27320</td>
      <td>0.26076</td>
      <td>0.32446</td>
      <td>0.381398</td>
      <td>0.373424</td>
      <td>0.195709</td>
      <td>0.774425</td>
      <td>3005.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.440945</td>
      <td>0.391128</td>
      <td>0.31796</td>
      <td>0.32128</td>
      <td>0.44467</td>
      <td>0.327915</td>
      <td>0.321570</td>
      <td>0.605077</td>
      <td>0.602642</td>
      <td>939.85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.178193</td>
      <td>0.247408</td>
      <td>0.24564</td>
      <td>0.22089</td>
      <td>0.21230</td>
      <td>0.204687</td>
      <td>0.202213</td>
      <td>0.246011</td>
      <td>0.432606</td>
      <td>2763.85</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 132 columns</p>
</div>



### Train test split


```python
from sklearn.model_selection import train_test_split
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
data_tr , data_te = train_test_split(data_df, test_size=0.2,shuffle=True,random_state=20)

X_train = data_tr.drop(['id', 'loss'], axis=1)
y_train = data_tr['loss'].values

X_test = data_te.drop(['id', 'loss'], axis=1)
y_test = data_te['loss'].values

print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('y_train shape: ', y_train.shape)
print('y_test shape: ', y_test.shape)
```

    X_train shape:  (150654, 130)
    X_test shape:  (37664, 130)
    y_train shape:  (150654,)
    y_test shape:  (37664,)
    

Question: Does training data have any null values?


```python
X_train.isnull().any().any()
```




    False



Training data have no null values.

### Visualizing target variable : loss

*Plotting the probability density function*


```python
plt.figure(figsize=(12,4))
plt.title('PDF plot of loss')
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot
sns.kdeplot(x='loss',data=data_tr)
plt.show()
```


![png](output_21_0.png)


- Distribution of target variable has a high right skew.
- Majority of data has loss below 10000.

*Plotting cumulative distribution function*


```python
plt.figure(figsize=(12,4))
plt.title('CDF plot of loss')
# https://seaborn.pydata.org/generated/seaborn.ecdfplot.html#seaborn.ecdfplot
sns.ecdfplot(x='loss',data=data_tr)
plt.show()
print('Minimim value of loss: ',y_train.min())
print('Maximum value of loss: ',y_train.max())

print('95 percentile of loss: %.2f'%np.percentile(y_train,95))
```


![png](output_24_0.png)


    Minimim value of loss:  5.25
    Maximum value of loss:  121012.25
    95 percentile of loss: 8487.69
    

*Creating two data frames one having categorical features and other having numerical features*


```python
cat_feature_df = X_train.select_dtypes(include='object')
num_feature_df = X_train.select_dtypes(include='float')
print('Catgorical features count: ',cat_feature_df.shape[1])
print('Numerical features count: ',num_feature_df.shape[1])
```

    Catgorical features count:  116
    Numerical features count:  14
    

**Statistics for numerical features**


```python
num_feature_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cont1</th>
      <th>cont2</th>
      <th>cont3</th>
      <th>cont4</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
      <td>150654.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.494049</td>
      <td>0.507435</td>
      <td>0.499103</td>
      <td>0.491998</td>
      <td>0.487148</td>
      <td>0.491374</td>
      <td>0.485165</td>
      <td>0.487029</td>
      <td>0.485743</td>
      <td>0.498448</td>
      <td>0.493673</td>
      <td>0.493330</td>
      <td>0.493635</td>
      <td>0.495807</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.187799</td>
      <td>0.207196</td>
      <td>0.202258</td>
      <td>0.211309</td>
      <td>0.209040</td>
      <td>0.205303</td>
      <td>0.178342</td>
      <td>0.199448</td>
      <td>0.181963</td>
      <td>0.185968</td>
      <td>0.209756</td>
      <td>0.209442</td>
      <td>0.212819</td>
      <td>0.222508</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000016</td>
      <td>0.001149</td>
      <td>0.002634</td>
      <td>0.176921</td>
      <td>0.281143</td>
      <td>0.012683</td>
      <td>0.069503</td>
      <td>0.236880</td>
      <td>0.000080</td>
      <td>0.000000</td>
      <td>0.035321</td>
      <td>0.036232</td>
      <td>0.000228</td>
      <td>0.180268</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.347403</td>
      <td>0.358319</td>
      <td>0.336963</td>
      <td>0.327354</td>
      <td>0.281143</td>
      <td>0.336105</td>
      <td>0.350175</td>
      <td>0.317960</td>
      <td>0.358970</td>
      <td>0.364580</td>
      <td>0.310961</td>
      <td>0.314945</td>
      <td>0.315758</td>
      <td>0.294758</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.475784</td>
      <td>0.555782</td>
      <td>0.527991</td>
      <td>0.452887</td>
      <td>0.422268</td>
      <td>0.440945</td>
      <td>0.438771</td>
      <td>0.441060</td>
      <td>0.437310</td>
      <td>0.461190</td>
      <td>0.457203</td>
      <td>0.462286</td>
      <td>0.363547</td>
      <td>0.408570</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.623912</td>
      <td>0.681761</td>
      <td>0.634224</td>
      <td>0.652072</td>
      <td>0.635304</td>
      <td>0.655553</td>
      <td>0.591045</td>
      <td>0.623580</td>
      <td>0.568890</td>
      <td>0.619840</td>
      <td>0.678924</td>
      <td>0.679096</td>
      <td>0.689974</td>
      <td>0.724610</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.984975</td>
      <td>0.862654</td>
      <td>0.944251</td>
      <td>0.952482</td>
      <td>0.983674</td>
      <td>0.997162</td>
      <td>1.000000</td>
      <td>0.980200</td>
      <td>0.995400</td>
      <td>0.994980</td>
      <td>0.998742</td>
      <td>0.998484</td>
      <td>0.988494</td>
      <td>0.844848</td>
    </tr>
  </tbody>
</table>
</div>



- Numerical features lies between 0 and 1
- Mean of all features close to 0.5
- Standard deviation of all features is about 0.2

**Statistics for categorical features**


```python
#https://www.kaggle.com/nextbigwhat/eda-for-categorical-variables-part-2
cat_stats = pd.DataFrame(
    columns=['column', 'unique_values', 'unique_value_count', 'nan_count'])

for c in cat_feature_df.columns:
    tmp = pd.DataFrame()
    tmp['column'] = [c]
    tmp['unique_values'] = [cat_feature_df[c].unique()]
    tmp['unique_value_count'] = int(cat_feature_df[c].nunique())
    tmp['nan_count'] = cat_feature_df[c].isnull().sum()
    cat_stats = cat_stats.append(tmp, ignore_index=True)
cat_stats.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>unique_values</th>
      <th>unique_value_count</th>
      <th>nan_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cat1</td>
      <td>[B, A]</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat2</td>
      <td>[A, B]</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat3</td>
      <td>[A, B]</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat4</td>
      <td>[A, B]</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat5</td>
      <td>[B, A]</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Above data frame holds the name, unique values and unique values count for every categorical feature. 

**Plotting a cdf plot of unique value counts of categorical features**


```python
plt.figure(figsize=(12,4))
plt.title('CDF of unique_value_count')
# https://seaborn.pydata.org/generated/seaborn.ecdfplot.html#seaborn.ecdfplot
sns.ecdfplot(x='unique_value_count' ,data=cat_stats)
print('95 percentile of unique_value_count: %d ' %np.percentile(cat_stats['unique_value_count'],95))
```

    95 percentile of unique_value_count: 20 
    


![png](output_34_1.png)


Only 5 percent features have cardinality more than 20.

### Encoding of categorical features

- *Since we have total 116 categorical features, it is very hard to perform EDA on each of them. We will pick 15 most important features which help to predict that target variable using feature selection technique.*
- *To apply feature selection technique it is required to encode our categorical features to numerical values.*

**Picking features for one hot encoding**


```python
cat_stat_lt5 = cat_stats[cat_stats['unique_value_count']<5]
one_hot_cols = cat_stat_lt5['column'].values
print('Categorical fetaure for one hot encoding: \n',one_hot_cols)
```

    Categorical fetaure for one hot encoding: 
     ['cat1' 'cat2' 'cat3' 'cat4' 'cat5' 'cat6' 'cat7' 'cat8' 'cat9' 'cat10'
     'cat11' 'cat12' 'cat13' 'cat14' 'cat15' 'cat16' 'cat17' 'cat18' 'cat19'
     'cat20' 'cat21' 'cat22' 'cat23' 'cat24' 'cat25' 'cat26' 'cat27' 'cat28'
     'cat29' 'cat30' 'cat31' 'cat32' 'cat33' 'cat34' 'cat35' 'cat36' 'cat37'
     'cat38' 'cat39' 'cat40' 'cat41' 'cat42' 'cat43' 'cat44' 'cat45' 'cat46'
     'cat47' 'cat48' 'cat49' 'cat50' 'cat51' 'cat52' 'cat53' 'cat54' 'cat55'
     'cat56' 'cat57' 'cat58' 'cat59' 'cat60' 'cat61' 'cat62' 'cat63' 'cat64'
     'cat65' 'cat66' 'cat67' 'cat68' 'cat69' 'cat70' 'cat71' 'cat72' 'cat73'
     'cat74' 'cat75' 'cat76' 'cat77' 'cat78' 'cat79' 'cat80' 'cat81' 'cat82'
     'cat83' 'cat84' 'cat85' 'cat86' 'cat87' 'cat88']
    

- We have picked the features having cardinality less than 5 for one hot encoding.
- One hot encoding is suitable if cardinality of feature is low

**Plotting bar plot for features having cardinality(c) more than or equal to 5**


```python
plt.figure(figsize=(18,4))
plt.title('Bar plot for unique_value_count vs coulmns (c>=5)')
cat_stat_ge5 = cat_stats[cat_stats['unique_value_count']>=5]
# https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot
sns.barplot(x='column',y='unique_value_count',data=cat_stat_ge5)
plt.show()
```


![png](output_42_0.png)


- Most of the columns have cardinality less than 50. 
- cat109, cat110, cat112, cat113, cat116 have cardinality more than or equal to 50.
- cat116 has very high cardinality which is more than 300.

**Picking features for binary encoding**


```python
cat_stat_5t32 = cat_stats[(cat_stats['unique_value_count'] >=5) & (cat_stats['unique_value_count'] <=32)]
binary_encoding_cols = cat_stat_5t32['column'].values
print('Categorical fetaure for binary encoding:\n',binary_encoding_cols)
```

    Categorical fetaure for binary encoding:
     ['cat89' 'cat90' 'cat91' 'cat92' 'cat93' 'cat94' 'cat95' 'cat96' 'cat97'
     'cat98' 'cat99' 'cat100' 'cat101' 'cat102' 'cat103' 'cat104' 'cat105'
     'cat106' 'cat107' 'cat108' 'cat111' 'cat114' 'cat115']
    

- Binary encoding encode the feature with cardinality 'c' into ceil(log2(c)) number of features.
- Here binary encoding will result in features of length 3 to 5.

**Picking features for target encoding**


```python
cat_stat_gt32 = cat_stats[cat_stats['unique_value_count']>32]
target_encoding_cols = cat_stat_gt32['column'].values
print('Categorical fetaure for target encoding: ',target_encoding_cols)
```

    Categorical fetaure for target encoding:  ['cat109' 'cat110' 'cat112' 'cat113' 'cat116']
    

- Features with high cardinality are picked for target encoding
- Target encoding is prone to over-fitting. Hence we will take care of regularization.

*Applying one hot encoding to one_hot_cols*


```python
one_hot_encoder = OneHotEncoder(cols=one_hot_cols)
one_hot_encoder.fit(X_train)
X_train_one_hot = one_hot_encoder.transform(X_train)
```

*Applying binary encoding to binary_encoding_cols*


```python
bin_encoder = BinaryEncoder(cols=binary_encoding_cols)
bin_encoder.fit(X_train_one_hot)
X_train_bin = bin_encoder.transform(X_train_one_hot)
```

*Applying target encoding to target_encoding_cols*


```python
y_train_tfmd = np.log(y_train)
target_encoder = TargetEncoder(
    cols=target_encoding_cols, min_samples_leaf=10, smoothing=10.0)
target_encoder.fit(X_train_bin, y_train_tfmd)
X_train_tar = target_encoder.transform(X_train_bin)
```

**Data frame after categorical encoding**


```python
X_train_tar.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cat1_1</th>
      <th>cat1_2</th>
      <th>cat2_1</th>
      <th>cat2_2</th>
      <th>cat3_1</th>
      <th>cat3_2</th>
      <th>cat4_1</th>
      <th>cat4_2</th>
      <th>cat5_1</th>
      <th>cat5_2</th>
      <th>...</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69757</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0.281143</td>
      <td>0.655021</td>
      <td>0.481161</td>
      <td>0.45883</td>
      <td>0.76280</td>
      <td>0.51111</td>
      <td>0.682315</td>
      <td>0.669033</td>
      <td>0.723122</td>
      <td>0.679136</td>
    </tr>
    <tr>
      <th>77241</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0.281143</td>
      <td>0.829824</td>
      <td>0.757347</td>
      <td>0.82598</td>
      <td>0.58325</td>
      <td>0.79863</td>
      <td>0.784967</td>
      <td>0.772574</td>
      <td>0.862949</td>
      <td>0.701420</td>
    </tr>
    <tr>
      <th>59852</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0.551723</td>
      <td>0.456946</td>
      <td>0.589494</td>
      <td>0.31280</td>
      <td>0.44352</td>
      <td>0.53328</td>
      <td>0.771508</td>
      <td>0.758883</td>
      <td>0.336261</td>
      <td>0.313226</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 333 columns</p>
</div>



### Selection of top features

*We are using DecisionTreeRegressor for feature selection*


```python
from sklearn.tree import DecisionTreeRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
model = DecisionTreeRegressor(random_state=10)
model.fit(X_train_tar, y_train_tfmd)
importance = model.feature_importances_
feature_score_dict = dict(zip(X_train_tar.columns.values, importance))
feature_score_dict = dict(
    sorted(feature_score_dict.items(), key=lambda x: x[1], reverse=True))
```

- DecisionTreeRegressor calculates the feature importance based on information gain due to a particular feature.
- DecisionTreeRegressor result is not affected by non-linear features.

**Plotting the feature importance CDF**


```python
plt.figure(figsize=(12,4))
plt.title('score vs features')
plt.plot(list(feature_score_dict.values()))
plt.xlabel('features')
plt.ylabel('scores')
plt.show()
```


![png](output_63_0.png)


*We will sort feature importance scores in descending order and pick the top 15 features.*


```python
plt.figure(figsize=(15, 4))
plt.title('Feature importance score for top 15 features')
# https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot
sns.barplot(
    x=list(feature_score_dict.keys())[:15], y=list(feature_score_dict.values())[:15])
plt.xlabel('Features')
plt.ylabel('Score')
plt.show()
```


![png](output_65_0.png)


- cat80 has highest feature importance score.
- Among top 15 features, 7 are categorical and 8 are numerical.

*We will perform EDA on original (non-transformed) features based on the top 15 features*


```python
original_features = list(e.split('_')[0]  for e in list(feature_score_dict.keys())[:15])
top_cat_features = [e for e in original_features if 'cat' in e]
top_num_features = [e for e in original_features if 'cont' in e]

print('Top 15 original features: \n', original_features)
print('Top categorical features: ', top_cat_features)
print('Top numerical features: ', top_num_features)
```

    Top 15 original features: 
     ['cat80', 'cat12', 'cont14', 'cont7', 'cont2', 'cat112', 'cat79', 'cat81', 'cat116', 'cat100', 'cont8', 'cont5', 'cont6', 'cont13', 'cont3']
    Top categorical features:  ['cat80', 'cat12', 'cat112', 'cat79', 'cat81', 'cat116', 'cat100']
    Top numerical features:  ['cont14', 'cont7', 'cont2', 'cont8', 'cont5', 'cont6', 'cont13', 'cont3']
    


```python
def cat_order(category):
    return data_tr[category].value_counts().keys()
```

### EDA on categorical features


```python

def plot_categorical(feature):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Plots for {}'.format(feature))
    ax[0].set_title('count vs category plot')
#     https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot
    sns.countplot(x=feature, data=data_tr, order=cat_order(feature), ax=ax[0])
    ax[1].set_title('loss vs category bar plot')
#     https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot
    sns.barplot(x=feature, y='loss', data=data_tr, order=cat_order(feature), estimator=np.mean, ax=ax[1])
    ax[2].set_title('loss vs category box plot')
#     https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot
    sns.boxplot(x=feature, y='loss', data=data_tr, order=cat_order(feature), ax=ax[2])
    plt.show()
```


```python
plot_categorical('cat80')
```


![png](output_72_0.png)


- Frequency of categories in unbalanced for cat80.
- D has major count and it corresponds to mean loss of 2000.
- Mean loss against each category of cat80 lies between 2000 and 6000


```python
plot_categorical('cat12')
```


![png](output_74_0.png)


- Count of B category is much high as compared to all others.
- B corresponds to mean loss of 2500. Since count of B very high corresponding loss will contribute the model more.
- All categories have high no of outliers of mean loss.


```python
plot_categorical('cat79')
```


![png](output_76_0.png)


- Count of A category is much high as compared to B.
- B corresponds to mean loss of 2500. Since count of B very high corresponding loss will contribute the model more.
- Both categories have high no of outliers of mean loss.


```python
plot_categorical('cat81')
```


![png](output_78_0.png)


- Count of D category is much high as compared to others.
- D corresponds to mean loss of 2500. Since count of D very high corresponding loss will contribute the model more.
- All categories have high no of outliers of mean loss.

**Comparing count vs category plot of above features**


```python
fig, ax = plt.subplots(1, 4, sharey=True, figsize=(18, 4))
# y-axis is shared among plots
fig.suptitle('Count vs Features')
categories=['cat80', 'cat12', 'cat79', 'cat81']
for i in range(4):
    sns.countplot(x=categories[i], data=data_tr, order=cat_order(categories[i]),ax=ax[i])
plt.show()
```


![png](output_81_0.png)


- Count of values is among all features is highly unbalanced. 
- One category dominates all others.

**Comparing loss vs category bar plot of above features**


```python
fig, ax = plt.subplots(1, 4, sharey=True, figsize=(18, 4))
fig.suptitle('Bar plot for grouped target mean')
# https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot
categories = ['cat80', 'cat12', 'cat79', 'cat81']
for i in range(4):
    sns.barplot(x=categories[i], y='loss', data=data_tr,
                estimator=np.mean, order=cat_order(categories[i]), ax=ax[i])
plt.show()
```


![png](output_84_0.png)


- Mean loss vs category is unbalanced which is good thing for model.
- Mean loss corresponding to highly frequent categories lies around 2000. Hence they point to lower loss.
- Less frequent categories corresponds to higher loss.

**Comparing loss vs category box plot of above features**


```python
fig, ax = plt.subplots(1, 4, sharey=True, figsize=(18, 6))
fig.suptitle('Box plot for grouped target mean')
# https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot
categories = ['cat80', 'cat12', 'cat79', 'cat81']
for i in range(4):
    sns.boxplot(x=categories[i], y='loss', data=data_tr,
                order=cat_order(categories[i]), ax=ax[i])
plt.show()
```


![png](output_87_0.png)


- Loss values quantiles for high frequent categories lies a lower values.
- We can see lot of outliers with high loss in every feature.

*Since cat100, cat112 and cat116 has high cardinality we are plotting them seprately*

**Plotting for cat100**


```python
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
ax1.set_title('count vs category plot')
sns.countplot(x='cat100',data=data_tr,order=cat_order('cat100'),ax=ax1)
sns.barplot(x='cat100', y='loss',data=data_tr,order=cat_order('cat100'), estimator=np.mean,ax=ax2)
plt.show()
```


![png](output_91_0.png)


- Frequency distribution of categories has variance. Contributes in better prediction of loss.
- Categories with high frequency corresponds to lower loss.
- Less frequent categories corresponds to higher loss.

**Plotting for cat112**


```python
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,10))

ax1.set_title('count vs category plot')
# https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot
sns.countplot(x='cat112',data=data_tr,order=cat_order('cat112'),ax=ax1)

ax1.set_title('loss vs category plot')
# https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot
sns.barplot(x='cat112', y='loss',data=data_tr,order=cat_order('cat112'), estimator=np.mean,ax=ax2)
plt.show()
```


![png](output_94_0.png)


- Frequency distribution of categories has variance.
- Loss per category is somewhat uniform. Not useful for prediction.

**Plotting for cat116**


```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
vc = data_tr['cat116'].value_counts()
vc_key = vc[vc > 500].index
data_temp = data_tr[data_tr['cat116'].isin(vc_key)].copy()

ax1.set_title('count vs category plot')
# https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot
sns.countplot(x='cat116', data=data_temp, order=vc_key, ax=ax1)
ax2.set_title('loss vs category plot')
# https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot
sns.barplot(x='cat116', y='loss', data=data_temp,
            order=vc_key, estimator=np.mean, ax=ax2)
plt.show()
del data_temp
```


![png](output_97_0.png)


- Frequency distribution of categories have variance.
- Loss per category is somewhat uniform. Not useful for prediction.
- Few categories have very high loss.

*Function to get Cramers test values*


```python
import scipy.stats as ss
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
```

**Plotting co-relation heatmap for categorical features**


```python
categories = top_cat_features
corr_matrix = []
for cat1 in tqdm(categories):
    corr = []
    for cat2 in categories:
        corr.append(cramers_v(data_tr[cat1], data_tr[cat2]))
    corr_matrix.append(corr)
sns.heatmap(np.array(corr_matrix), annot=True, cmap='viridis', linewidths=2, fmt='.3f',
            xticklabels=categories, yticklabels=categories)
```

    100%|████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.68it/s]
    




    <AxesSubplot:>




![png](output_102_2.png)


- (cat80,cat81) (cat12,cat100) (cat80,cat79) have high co-relation.

### EDA on numerical features

**Analyzing distribution of numerical features**

*Here features are plotted in order of their feature importance*


```python
 print('top_num_features',top_num_features)
```

    top_num_features ['cont14', 'cont7', 'cont2', 'cont8', 'cont5', 'cont6', 'cont13', 'cont3']
    


```python
from matplotlib.colors import Colormap as cm
sns.set_style('whitegrid')
sns.set_palette("tab10")
plt.figure(figsize=(15,5))
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot
for i in range(len(top_num_features)):
    g = sns.kdeplot(x=top_num_features[i],data=data_tr,label=top_num_features[i])
plt.xlabel('feature') 

plt.legend()
plt.show()
```


![png](output_108_0.png)


- Most of the distribution is more like uniform.
- cont5, cont13 is highly non uniform with a skew. Hence its feature importance in low.

**Plotting the coorelation heatmap**


```python
cont_df = X_train[top_num_features].copy()
cont_df['loss'] = y_train_tfmd 
plt.figure(figsize=(9,7))
# https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap
sns.heatmap(cont_df.corr(),annot=True,cmap='Spectral',linewidths=2,fmt='.2f')
```




    <AxesSubplot:>




![png](output_111_1.png)


- (cont13,cont6) (cont6,cont7) has high co-relation.
- cont7, cont2  has higher co-relation with loss which supports their feature importance.
- cont14 has less co-relation with loss but its importance is high for some reason.

**Plotting regression plot among numerical features and loss**

- *Here we will do regression plot between continuous features and target loss*
- *Here our numerical feature are divided in bins. Loss is plotted in log-scale.*
- *Each vertical stick has mid point as mean and its length tell the confidence interval of the x_bin*
- *Faded region show the confidence interval of calculated n_boot samples*
- *Since number of data points is very large we will randomly pick 1/50th of total train points for our plots.*


```python
# https://seaborn.pydata.org/generated/seaborn.regplot.html#seaborn.regplot
max_num = data_tr.shape[0]
size = int(data_tr.shape[0]/50)
random_index = [int(i) for i in np.random.uniform(0, max_num, size)]

data_sample = data_tr.iloc[random_index, :]
sns.set_style('darkgrid')

fig, ax = plt.subplots(3, 3, figsize=(18, 15), sharey=True)

k = 0
for i in range(3):
    for j in range(3):
        if k == 7:
            break
        sns.regplot(x=top_num_features[k], y='loss', data=data_sample,
                    x_estimator=np.mean, x_bins=20, x_ci=95, n_boot=500, ax=ax[i][j])

        ax[i][j].set_yscale('log')
        ax[i][j].set_ylabel('loss in log-scale')
        k += 1
fig.tight_layout()
ax[2][1].remove()
ax[2][2].remove()
plt.show()
```


![png](output_115_0.png)


- Regression line for cont7, cont2 has nice positive slope which supports their higher co-relation result and their higher feature importance score.
- Regression line for con8, cont5 and cont13 is nearly flat which show that they have lower co-relation with target loss
- cont14 have less slope than cont7 and cont2 but its feature importance is more for some reason.

### Performing preprocessing on data

*Function for target transformation*


```python
class TargetTransform():
    def __init__(self, func):
        if func == 'fourth_root':
            self.func = lambda x: x**0.25
            self.func_inv = lambda x: x**4
        if func == 'log':
            self.func = lambda x: np.log(x)
            self.func_inv = lambda x: np.exp(x)

    def transform(self, target):
        return np.array([self.func(e) for e in target])

    def inverse_transform(self, target):
        return np.array([self.func_inv(e) for e in target])
```


```python
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
import pickle


std_scale=StandardScaler()
one_hot_encoder = OneHotEncoder(cols=one_hot_cols)
bin_encoder = BinaryEncoder(cols=binary_encoding_cols)
target_encoder = TargetEncoder(cols=target_encoding_cols, min_samples_leaf=10, smoothing=10.0)
scale = ColumnTransformer([('std_scale', std_scale, target_encoding_cols)],remainder='passthrough')

preprocessing = Pipeline([
    ('one_hot_encoder', one_hot_encoder),
    ('bin_encoder',bin_encoder),
    ('target_encoder',target_encoder),
    ('scale',scale)
])
target_tf = TargetTransform('log')
preprocessing.fit(X_train,target_tf.transform(y_train))

pickle.dump(preprocessing,open('preprocessing.pkl','wb'))

X_train_prsd=preprocessing.transform(X_train)
X_test_prsd=preprocessing.transform(X_test)
```

### Performing feature extraction

- *We are using auto-encoder for extracting 50 additional features from training data*
- *We will stack these addition features along with processed features.*


```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger

# https://machinelearningmastery.com/autoencoder-for-regression/
n_inputs = X_train_prsd.shape[1]
# define bottleneck
n_bottleneck = 50

visible = Input(shape=(n_inputs,))
e = Dense(n_inputs//2)(visible)
e = BatchNormalization()(e)
e = ReLU()(e)
bottleneck = Dense(n_bottleneck)(e)
# define decoder
d = Dense(n_inputs//2)(bottleneck)
d = BatchNormalization()(d)
d = ReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')
# plot the autoencoder
plot_model(model, 'autoencoder.png', show_shapes=True)
# fit the autoencoder model to reconstruct input
csv_logger = CSVLogger('train_log.csv', separator=',', append=False)
history = model.fit(X_train_prsd, X_train_prsd, epochs=100, batch_size=32,
                    verbose=2, validation_data=(X_test_prsd, X_test_prsd),
                    callbacks=[csv_logger])
```

    Epoch 1/100
    4708/4708 - 13s - loss: 0.0379 - val_loss: 0.0188
    Epoch 2/100
    4708/4708 - 11s - loss: 0.0164 - val_loss: 0.0132
    Epoch 3/100
    4708/4708 - 10s - loss: 0.0140 - val_loss: 0.0111
    Epoch 4/100
    4708/4708 - 10s - loss: 0.0130 - val_loss: 0.0112
    Epoch 5/100
    4708/4708 - 11s - loss: 0.0123 - val_loss: 0.0100
    Epoch 6/100
    4708/4708 - 10s - loss: 0.0118 - val_loss: 0.0094
    Epoch 7/100
    4708/4708 - 10s - loss: 0.0113 - val_loss: 0.0089
    Epoch 8/100
    4708/4708 - 10s - loss: 0.0109 - val_loss: 0.0087
    Epoch 9/100
    4708/4708 - 10s - loss: 0.0106 - val_loss: 0.0086
    Epoch 10/100
    4708/4708 - 10s - loss: 0.0104 - val_loss: 0.0082
    Epoch 11/100
    4708/4708 - 10s - loss: 0.0102 - val_loss: 0.0080
    Epoch 12/100
    4708/4708 - 10s - loss: 0.0101 - val_loss: 0.0083
    Epoch 13/100
    4708/4708 - 11s - loss: 0.0099 - val_loss: 0.0075
    Epoch 14/100
    4708/4708 - 10s - loss: 0.0097 - val_loss: 0.0075
    Epoch 15/100
    4708/4708 - 10s - loss: 0.0096 - val_loss: 0.0073
    Epoch 16/100
    4708/4708 - 10s - loss: 0.0095 - val_loss: 0.0072
    Epoch 17/100
    4708/4708 - 11s - loss: 0.0095 - val_loss: 0.0072
    Epoch 18/100
    4708/4708 - 10s - loss: 0.0094 - val_loss: 0.0070
    Epoch 19/100
    4708/4708 - 11s - loss: 0.0093 - val_loss: 0.0070
    Epoch 20/100
    4708/4708 - 10s - loss: 0.0092 - val_loss: 0.0068
    Epoch 21/100
    4708/4708 - 11s - loss: 0.0092 - val_loss: 0.0068
    Epoch 22/100
    4708/4708 - 12s - loss: 0.0091 - val_loss: 0.0071
    Epoch 23/100
    4708/4708 - 11s - loss: 0.0090 - val_loss: 0.0068
    Epoch 24/100
    4708/4708 - 10s - loss: 0.0089 - val_loss: 0.0063
    Epoch 25/100
    4708/4708 - 14s - loss: 0.0089 - val_loss: 0.0066
    Epoch 26/100
    4708/4708 - 14s - loss: 0.0089 - val_loss: 0.0065
    Epoch 27/100
    4708/4708 - 11s - loss: 0.0088 - val_loss: 0.0064
    Epoch 28/100
    4708/4708 - 11s - loss: 0.0088 - val_loss: 0.0065
    Epoch 29/100
    4708/4708 - 11s - loss: 0.0087 - val_loss: 0.0064
    Epoch 30/100
    4708/4708 - 11s - loss: 0.0087 - val_loss: 0.0064
    Epoch 31/100
    4708/4708 - 11s - loss: 0.0087 - val_loss: 0.0062
    Epoch 32/100
    4708/4708 - 11s - loss: 0.0086 - val_loss: 0.0063
    Epoch 33/100
    4708/4708 - 14s - loss: 0.0086 - val_loss: 0.0062
    Epoch 34/100
    4708/4708 - 13s - loss: 0.0085 - val_loss: 0.0062
    Epoch 35/100
    4708/4708 - 11s - loss: 0.0085 - val_loss: 0.0064
    Epoch 36/100
    4708/4708 - 11s - loss: 0.0085 - val_loss: 0.0060
    Epoch 37/100
    4708/4708 - 11s - loss: 0.0084 - val_loss: 0.0061
    Epoch 38/100
    4708/4708 - 11s - loss: 0.0084 - val_loss: 0.0061
    Epoch 39/100
    4708/4708 - 10s - loss: 0.0084 - val_loss: 0.0063
    Epoch 40/100
    4708/4708 - 10s - loss: 0.0084 - val_loss: 0.0062
    Epoch 41/100
    4708/4708 - 10s - loss: 0.0084 - val_loss: 0.0059
    Epoch 42/100
    4708/4708 - 10s - loss: 0.0083 - val_loss: 0.0059
    Epoch 43/100
    4708/4708 - 10s - loss: 0.0083 - val_loss: 0.0058
    Epoch 44/100
    4708/4708 - 10s - loss: 0.0083 - val_loss: 0.0061
    Epoch 45/100
    4708/4708 - 10s - loss: 0.0082 - val_loss: 0.0058
    Epoch 46/100
    4708/4708 - 10s - loss: 0.0082 - val_loss: 0.0058
    Epoch 47/100
    4708/4708 - 10s - loss: 0.0082 - val_loss: 0.0057
    Epoch 48/100
    4708/4708 - 10s - loss: 0.0082 - val_loss: 0.0062
    Epoch 49/100
    4708/4708 - 12s - loss: 0.0082 - val_loss: 0.0058
    Epoch 50/100
    4708/4708 - 11s - loss: 0.0082 - val_loss: 0.0056
    Epoch 51/100
    4708/4708 - 11s - loss: 0.0082 - val_loss: 0.0059
    Epoch 52/100
    4708/4708 - 11s - loss: 0.0081 - val_loss: 0.0058
    Epoch 53/100
    4708/4708 - 10s - loss: 0.0081 - val_loss: 0.0057
    Epoch 54/100
    4708/4708 - 10s - loss: 0.0081 - val_loss: 0.0057
    Epoch 55/100
    4708/4708 - 10s - loss: 0.0081 - val_loss: 0.0059
    Epoch 56/100
    4708/4708 - 10s - loss: 0.0081 - val_loss: 0.0057
    Epoch 57/100
    4708/4708 - 10s - loss: 0.0080 - val_loss: 0.0058
    Epoch 58/100
    4708/4708 - 11s - loss: 0.0080 - val_loss: 0.0056
    Epoch 59/100
    4708/4708 - 10s - loss: 0.0080 - val_loss: 0.0055
    Epoch 60/100
    4708/4708 - 10s - loss: 0.0080 - val_loss: 0.0057
    Epoch 61/100
    4708/4708 - 10s - loss: 0.0080 - val_loss: 0.0056
    Epoch 62/100
    4708/4708 - 10s - loss: 0.0079 - val_loss: 0.0057
    Epoch 63/100
    4708/4708 - 10s - loss: 0.0080 - val_loss: 0.0055
    Epoch 64/100
    4708/4708 - 10s - loss: 0.0079 - val_loss: 0.0057
    Epoch 65/100
    4708/4708 - 12s - loss: 0.0080 - val_loss: 0.0055
    Epoch 66/100
    4708/4708 - 11s - loss: 0.0079 - val_loss: 0.0055
    Epoch 67/100
    4708/4708 - 10s - loss: 0.0079 - val_loss: 0.0055
    Epoch 68/100
    4708/4708 - 11s - loss: 0.0079 - val_loss: 0.0056
    Epoch 69/100
    4708/4708 - 12s - loss: 0.0079 - val_loss: 0.0054
    Epoch 70/100
    4708/4708 - 11s - loss: 0.0079 - val_loss: 0.0055
    Epoch 71/100
    4708/4708 - 10s - loss: 0.0078 - val_loss: 0.0054
    Epoch 72/100
    4708/4708 - 10s - loss: 0.0078 - val_loss: 0.0055
    Epoch 73/100
    4708/4708 - 10s - loss: 0.0078 - val_loss: 0.0053
    Epoch 74/100
    4708/4708 - 10s - loss: 0.0078 - val_loss: 0.0055
    Epoch 75/100
    4708/4708 - 10s - loss: 0.0078 - val_loss: 0.0053
    Epoch 76/100
    4708/4708 - 10s - loss: 0.0078 - val_loss: 0.0053
    Epoch 77/100
    4708/4708 - 11s - loss: 0.0078 - val_loss: 0.0055
    Epoch 78/100
    4708/4708 - 10s - loss: 0.0078 - val_loss: 0.0052
    Epoch 79/100
    4708/4708 - 10s - loss: 0.0078 - val_loss: 0.0053
    Epoch 80/100
    4708/4708 - 11s - loss: 0.0078 - val_loss: 0.0053
    Epoch 81/100
    4708/4708 - 10s - loss: 0.0077 - val_loss: 0.0054
    Epoch 82/100
    4708/4708 - 10s - loss: 0.0077 - val_loss: 0.0053
    Epoch 83/100
    4708/4708 - 10s - loss: 0.0077 - val_loss: 0.0053
    Epoch 84/100
    4708/4708 - 10s - loss: 0.0077 - val_loss: 0.0053
    Epoch 85/100
    4708/4708 - 10s - loss: 0.0077 - val_loss: 0.0054
    Epoch 86/100
    4708/4708 - 10s - loss: 0.0077 - val_loss: 0.0053
    Epoch 87/100
    4708/4708 - 10s - loss: 0.0076 - val_loss: 0.0052
    Epoch 88/100
    4708/4708 - 11s - loss: 0.0076 - val_loss: 0.0052
    Epoch 89/100
    4708/4708 - 12s - loss: 0.0077 - val_loss: 0.0053
    Epoch 90/100
    4708/4708 - 11s - loss: 0.0076 - val_loss: 0.0053
    Epoch 91/100
    4708/4708 - 11s - loss: 0.0076 - val_loss: 0.0053
    Epoch 92/100
    4708/4708 - 11s - loss: 0.0076 - val_loss: 0.0051
    Epoch 93/100
    4708/4708 - 11s - loss: 0.0076 - val_loss: 0.0052
    Epoch 94/100
    4708/4708 - 11s - loss: 0.0076 - val_loss: 0.0052
    Epoch 95/100
    4708/4708 - 10s - loss: 0.0076 - val_loss: 0.0051
    Epoch 96/100
    4708/4708 - 10s - loss: 0.0076 - val_loss: 0.0052
    Epoch 97/100
    4708/4708 - 10s - loss: 0.0075 - val_loss: 0.0053
    Epoch 98/100
    4708/4708 - 10s - loss: 0.0075 - val_loss: 0.0051
    Epoch 99/100
    4708/4708 - 10s - loss: 0.0075 - val_loss: 0.0051
    Epoch 100/100
    4708/4708 - 10s - loss: 0.0075 - val_loss: 0.0051
    


```python
# plot loss
train_log = pd.read_csv('train_log.csv')
plt.plot(train_log['loss'], label='train')
plt.plot(train_log['val_loss'], label='test')
plt.title('Loss vs Epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
```


![png](output_124_0.png)


*Saving the encoder model*


```python
# define an encoder model (without the decoder)
if not path.exists('encoder.h5'):
    encoder = Model(inputs=visible, outputs=bottleneck)
    # save the encoder to file
    encoder.save('encoder.h5')
```

*Getting encoded features from auto-encoder and stack them with previous features*


```python
from tensorflow.keras.models import load_model
encoder = load_model('encoder.h5')
# encode the train data
X_train_encode = encoder.predict(X_train_prsd)
# encode the test data
X_test_encode = encoder.predict(X_test_prsd)
X_train_final = np.hstack([X_train_prsd,X_train_encode])
X_test_final = np.hstack([X_test_prsd,X_test_encode])
```

    WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
    

*Loding saved train and test data*


```python
if not path.exists('X_train_final.npy'):
    np.save('X_train_final.npy', X_train_final)
else:
    X_train_final = np.load('X_train_final.npy')

if not path.exists('X_test_final.npy'):
    np.save('X_test_final.npy', X_test_final)
else:
    X_test_final = np.load('X_test_final.npy')
    
if not path.exists('y_train.npy'):
    np.save('y_train.npy', y_train)
else:
    y_train = np.load('y_train.npy')
    
if not path.exists('y_test.npy'):
    np.save('y_test.npy', y_test)
else:
    y_test = np.load('y_test.npy')
```

### Building base line model

*Our base model will give random out within min and max range of target data*


```python
target_tf = TargetTransform('log')
min_y = y_train.min()
max_y = y_train.max()
y_train_predict = target_tf.transform(np.random.uniform(min_y,max_y,y_train.shape[0]))
y_test_predict = target_tf.transform(np.random.uniform(min_y,max_y,y_test.shape[0]))
print('Train mean_absolute_error: ', mean_absolute_error(
    y_train, target_tf.inverse_transform(y_train_predict)))
print('Test mean_absolute_error: ', mean_absolute_error(
    y_test, target_tf.inverse_transform(y_test_predict)))
```

    Train mean_absolute_error:  57504.66834075032
    Test mean_absolute_error:  57650.14391107142
    

### Finding and key take aways

- Dataset has mix of categorical and numerical features.
- There are very large no of categorical features as compared to numerical features.
- Dataset feature label has no real world meanings
- Dataset has no null values 
- Target variable is has as pareto distribution.
- Different categorical features have different cardinality.
- Based on the cardinality applied one-hot, binary and target encoding for categorical features.
- Each categorical features has fews categories dominating in number compared to others. These categories point towards lower values of loss
- Although there are very few continuous feature in dataset but feature importance is high.
- Most continuous features have uniform like distribution.
- We performed log-transform on target variable to make it uniform-like distribution.
- We applied standard scaler on our target encoder feature.
- We used auto-encoder for creating new features. 50 newly created features added to existing features.
- Our base-line model is a random number generating model.

### Models creation

#### SGDRegressor


```python
from sklearn.linear_model import SGDRegressor
target_tf = TargetTransform('log')

model = SGDRegressor(loss='squared_loss', penalty='l2',
                     alpha=10, random_state=10)

model.fit(X_train_final, target_tf.transform(y_train))
joblib.dump(model, 'models/sgd_regressor.sav')
y_train_predict = model.predict(X_train_final)
y_test_predict = model.predict(X_test_final)

print('Train mean_absolute_error: ', mean_absolute_error(
    y_train, target_tf.inverse_transform(y_train_predict)))

print('Test mean_absolute_error: ', mean_absolute_error(
    y_test, target_tf.inverse_transform(y_test_predict)))
```

    Train mean_absolute_error:  2150.430410997826
    Test mean_absolute_error:  2173.4739191631675
    

#### Decision Tree Regressor


```python
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=12, 
                               min_samples_split=80, min_samples_leaf=180, 
                               max_features='auto',random_state=10 )

dt_reg.fit(X_train_final, np.log(y_train))
joblib.dump(dt_reg, 'models/dt_regressor.sav')

y_train_predict = dt_reg.predict(X_train_final)
y_test_predict = dt_reg.predict(X_test_final)

print('Train mean_absolute_error: ', mean_absolute_error(
    y_train, np.exp(y_train_predict)))
print('Test mean_absolute_error: ', mean_absolute_error(
    y_test, np.exp(y_test_predict)))
```

    Train mean_absolute_error:  1257.650738239111
    Test mean_absolute_error:  1299.4206045211301
    

#### RandomForestRegressor 


```python
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=200, criterion='mse', max_depth=14,
                              min_samples_split=50, min_samples_leaf=10,
                              n_jobs=-1, random_state=10,
                              max_samples=0.2)
rf_reg.fit(X_train_final,np.log(y_train))
joblib.dump(dt_reg, 'models/rf_regressor.sav')

y_train_predict = rf_reg.predict(X_train_final)
y_test_predict = rf_reg.predict(X_test_final)

print('Train mean_absolute_error: ', mean_absolute_error(
    y_train, np.exp(y_train_predict)))
print('Test mean_absolute_error: ', mean_absolute_error(
    y_test, np.exp(y_test_predict)))
```

    Train mean_absolute_error:  1186.7141073228051
    Test mean_absolute_error:  1246.8563610534904
    

### AdaBoost Regressor


```python
from sklearn.ensemble import AdaBoostRegressor
ad_reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=12,
                                                       random_state=10),
                  learning_rate=1, loss='exponential', n_estimators=150,
                  random_state=10)


ad_reg.fit(X_train_final, np.log(y_train))
joblib.dump(xgb, 'models/xgb_regressor.sav')

y_train_predict = ad_reg.predict(X_train_final)
y_test_predict = ad_reg.predict(X_test_final)

print('Train mean_absolute_error: ', mean_absolute_error(
    y_train, np.exp(y_train_predict)))
print('Test mean_absolute_error: ', mean_absolute_error(
    y_test, np.exp(y_test_predict)))

```

    Train mean_absolute_error:  1103.9124882818053
    Test mean_absolute_error:  1266.3588895186997
    

### Gradient Boosted Decision Trees


```python
xgb = XGBRegressor(objective='reg:squarederror',
                         n_estimators=200,
                         booster='gbtree',
                         learning_rate=0.1,
                         max_depth=6,
                         gamma=0.1,
                         subsample=0.8,
                         min_child_weight=9,
                         colsample_bytree=0.7,
                         random_state=10,
                         n_jobs=-1)
xgb.fit(X_train_final, np.log(y_train))
joblib.dump(xgb, 'models/xgb_regressor.sav')

y_train_predict = xgb.predict(X_train_final)
y_test_predict = xgb.predict(X_test_final)

print('Train mean_absolute_error: ', mean_absolute_error(
    y_train, np.exp(y_train_predict)))
print('Test mean_absolute_error: ', mean_absolute_error(
    y_test, np.exp(y_test_predict)))
```

    Train mean_absolute_error:  1085.340836442061
    Test mean_absolute_error:  1173.694805258376
    
