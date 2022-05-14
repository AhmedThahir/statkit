# Statkit
Supplement your sci-kit learn models with 95 % confidence intervals and p-values.

## Description
- Estimate 95 % confidence intervals for your test scores.

For example, to compute a 95 % confidence interval of the area under the
receiver operating characteristic curve (ROC AUC):
```python
from sklearn.metrics import roc_auc_score
from statkit.non_parametric import bootstrap_score
from statkit.views import format_confidence_interval

y_prob = model.predict(X_test)
auc_95ci: dict = bootstrap_score(y_test, y_prob, metric=roc_auc_score)
print('Area under the ROC curve:', format_confidence_interval(auc_95ci, latex=False))
```

- Compute p-value to test if one model is significantly better than another.

For example, to test if the area under the receiver operating characteristic
curve (ROC AUC) of model 1 is significantly larger than model 2:
```python
from sklearn.metrics import roc_auc_score
from statkit.non_parametric import paired_permutation_test

y_pred_1 = model_1.predict(X_test)
y_pred_2 = model_2.predict(X_test)
_, p_value = paired_permutation_test(y_test, y_pred_1, y_pred_2, metric=roc_auc_score)
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
