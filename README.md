# Dtreehub

[![Stars](https://github.com/tkhan11/Dtreehub)
[![License]( https://github.com/tkhan11/Dtreehub/blob/master/LICENSE)

**Dtreehub** is a lightweight decision tree framework for Python **with categorical feature support**. It covers regular decision tree algorithms: ID3, C4.5, CART, CHAID and regression tree, random forest and adaboost. You just need to write **a few lines of code** to build decision trees with Dtreehub.


Basically, you just need to pass the dataset as pandas data frame and tree configurations optionally after importing Dtreehub as illustrated below. You just need to put the target label to the right. Besides, Dtreehub handles both numeric and nominal features and target values in contrast to its alternatives.

```python
from Dtreehub import Dtreehub as dtb
import pandas as pd

df = pd.read_csv("dataset/golf.txt")
config = {'algorithm': 'C4.5'}
model = dtb.fit(df, config = config)
```

**Outcomes**

Built decision trees are stored as python if statements in the `tests/outputs/rules` directory. A sample of decision rules is demonstrated below.

```python
def findDecision(Outlook, Temperature, Humidity, Wind):
   if Outlook == 'Rain':
      if Wind == 'Weak':
         return 'Yes'
      elif Wind == 'Strong':
         return 'No'
      else:
         return 'No'
   elif Outlook == 'Sunny':
      if Humidity == 'High':
         return 'No'
      elif Humidity == 'Normal':
         return 'Yes'
      else:
         return 'Yes'
   elif Outlook == 'Overcast':
      return 'Yes'
   else:
      return 'Yes'
 ```

**Testing for custom instances**

Decision rules will be stored in `outputs/rules/` folder when you build decision trees. You can run the built decision tree for new instances as illustrated below.

```python
prediction = dtb.predict(model, param = ['Sunny', 'Hot', 'High', 'Weak'])
```

You can consume built decision trees directly as well. In this way, you can restore already built decision trees and skip learning steps, or apply [transfer learning](https://youtu.be/9hX8ir7_ZtA). Loaded trees offer you findDecision method to test for new instances.

```python
moduleName = "outputs/rules/rules" #this will load outputs/rules/rules.py
tree = dtb.restoreTree(moduleName)
prediction = tree.findDecision(['Sunny', 'Hot', 'High', 'Weak'])
```

tests/global-unit-test.py will guide you how to build a different decision trees and make predictions.

**Model save and restoration**

You can save your trained models. This makes your model ready for transfer learning.

```python
dtb.save_model(model, "model.pkl")
```

In this way, you can use the same model later to just make predictions. This skips the training steps. Restoration requires to store .py and .pkl files under `outputs/rules`.

```python
model = dtb.load_model("model.pkl")
prediction = dtb.predict(model, ['Sunny',85,85,'Weak'])
```

### Sample configurations

Dtreehub supports several decision tree, bagging and boosting algorithms. You just need to pass the configuration to use different algorithms.

**Regular Decision Trees**

Regular decision tree algorithms find the best feature and the best split point maximizing the information gain. It builds decision trees recursively in child nodes.

```python
config = {'algorithm': 'C4.5'} #Set algorithm to ID3, C4.5, CART, CHAID or Regression
model = dtb.fit(df, config)
```

The following regular decision tree algorithms are wrapped in the library.

| Algorithm  | Metric | Tutorial | Demo |
| ---        | --- | ---      | ---  |
| ID3        | Entropy, Information Gain  |
| C4.5       | Entropy, Gain Ratio | |
| CART       | GINI | 
| CHAID      | Chi Square | 
| Regression | Standard Deviation | 

**Gradient Boosting**

Gradient boosting is basically based on building a tree, and then building another based on the previous one's error. In this way, it boosts results. Predictions will be the sum of each tree'e prediction result.

```python
config = {'enableGBM': True, 'epochs': 7, 'learning_rate': 1, 'max_depth': 5}
```

**Random Forest**

Random forest basically splits the data set into several sub data sets and builds different data set for those sub data sets. Predictions will be the average of each tree's prediction result.

```python
config = {'enableRandomForest': True, 'num_of_trees': 5}
```

**Adaboost**

Adaboost applies a decision stump instead of a decision tree. This is a weak classifier and aims to get min 50% score. It then increases the unclassified ones and decreases the classified ones. In this way, it aims to have a high score with weak classifiers.

```python
config = {'enableAdaboost': True, 'num_of_weak_classifier': 4}
```

**Feature Importance** 
Decision trees are naturally interpretable and explainable algorithms. A decision is clear made by a single tree. Still we need some extra layers to understand the built models. Besides, random forest and GBM are hard to explain.

```python
df = dtb.feature_importance("outputs/rules/rules.py")
```

| feature     | final_importance |
| ---         | ---              |
| Humidity    | 0.3688           |
| Wind        | 0.3688           |
| Outlook     | 0.2624           |
| Temperature | 0.0000           |

### Paralellism

Dtreehub offers parallelism to speed model building up. Branches of a decision tree will be created in parallel in this way. You should set enableParallelism argument to True in the configuration. Its default value is False. It allocates half of the total number of cores in your environment if parallelism is enabled.

```python
if __name__ == '__main__':
   config = {'algorithm': 'C4.5', 'enableParallelism': True, 'num_cores': 2}
   model = dtb.fit(df, config)
```

Notice that you have to locate training step in an if block and it should check you are in main.

### Contributing

Pull requests are welcome. You should run the unit tests locally by running [`test/global-unit-test.py`](https://github.com/tkhan11/Dtreehub/blob/master/tests/global-unit-test.py). Please share the unit test result logs in the PR.

### Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one.

### Licence

Dtreehub is licensed under the MIT License - see [`LICENSE`](https://github.com/tkhan11/Dtreehub/blob/master/LICENSE)
for more details.

