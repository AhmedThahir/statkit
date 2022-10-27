# Statkit
Supplement your sci-kit learn models with 95 % confidence intervals, p-values, and decision curves.

## Description
- Estimate 95 % confidence intervals for your test scores.

For example, to compute a 95 % confidence interval of the area under the
receiver operating characteristic curve (ROC AUC):
```python
from sklearn.metrics import roc_auc_score
from statkit.non_parametric import bootstrap_score

y_prob = model.predict_proba(X_test)[:, 1]
auc_95ci = bootstrap_score(y_test, y_prob, metric=roc_auc_score)
print('Area under the ROC curve:', auc_95ci)
```

- Compute p-value to test if one model is significantly better than another.

For example, to test if the area under the receiver operating characteristic
curve (ROC AUC) of model 1 is significantly larger than model 2:
```python
from sklearn.metrics import roc_auc_score
from statkit.non_parametric import paired_permutation_test

y_pred_1 = model_1.predict_proba(X_test)[:, 1]
y_pred_2 = model_2.predict_proba(X_test)[:, 1]
p_value = paired_permutation_test(y_test, y_pred_1, y_pred_2, metric=roc_auc_score)
```

- Perform decision curve analysis by making net benefit plots of your sci-kit learn models. Compare the utility of different models and with decision policies to always or never take an action/intervention.

![Net benefit curve](figures/demo_net_benefit_curve.png)
```python
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from statkit.decision import NetBenefitDisplay

centers = [[0, 0], [1, 1]]
X_train, y_train = make_blobs(
    centers=centers, cluster_std=1, n_samples=20, random_state=5
)
X_test, y_test = make_blobs(
    centers=centers, cluster_std=1, n_samples=20, random_state=1005
)

baseline_model = LogisticRegression(random_state=5).fit(X_train, y_train)
y_pred_base = baseline_model.predict_proba(X_test)[:, 1]

tree_model = GradientBoostingClassifier(random_state=5).fit(X_train, y_train)
y_pred_tree = tree_model.predict_proba(X_test)[:, 1]

NetBenefitDisplay.from_predictions(y_test, y_pred_base, name='Baseline model')
NetBenefitDisplay.from_predictions(y_test, y_pred_tree, name='Gradient boosted trees', show_references=False, ax=plt.gca())
```

Detailed documentation can be on the [Statkit API documentation pages](https://hylkedonker.gitlab.io/statkit).

## Installation
```bash
pip3 install statkit
```

## Support
You can open a ticket in the [Issue tracker](https://gitlab.com/hylkedonker/statkit/-/issues).

## Contributing
We are open for contributions.
If you open a pull request, make sure that your code is:
- Well documented,
- Code formatted with [black](https://github.com/psf/black),
- And contains an accompanying unit test.


## Authors and acknowledgment
Hylke C. Donker

## License
This code is licensed under the [MIT license](LICENSE).
