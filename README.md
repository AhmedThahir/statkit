# Statkit
Supplement your sci-kit learn models with 95 % confidence intervals and p-values.

## Description
- Estimate 95 % confidence intervals for your test scores.

For example, to compute a 95 % confidence interval of the area under the
receiver operating characteristic curve (ROC AUC):
```python
from sklearn.metrics import roc_auc_score
from statkit import non_parametric

y_prob = model.predict_proba(X_test)
non_parametric.bootstrap_score(y_test, y_prob[:, 1], metric=roc_auc_score)
```

- Compute p-value to test if one model is significantly better than another.

For example, to test if the area under the receiver operating characteristic
curve (ROC AUC) of model 1 is significantly larger than model 2:
```python
from sklearn.metrics import roc_auc_score
from statkit import non_parametric

y_pred_1 = model_1.predict_proba(X_test)
y_pred_2 = model_2.predict_proba(X_test)
, p_value = non_parametric.paired_permutation_test(
    y_test,
    y_pred_1[:, 1],
    y_pred_2[:, 1],
    metric=roc_auc_score,
)
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
