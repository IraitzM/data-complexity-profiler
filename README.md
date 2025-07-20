[![SQAaaS badge shields.io](https://github.com/EOSC-synergy/data-complexity-profiler.assess.sqaaas/raw/main/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/data-complexity-profiler.assess.sqaaas/main/.report/assessment_output.json)

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/IraitzM/data-complexity-profiler?utm_source=oss&utm_medium=github&utm_campaign=IraitzM%2Fdata-complexity-profiler&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

# Data Complexity

The Data Complexity Measures in pure Python.

## Install

```bash
pip install data-complexity-profiler
```

## How it works

One can import the model and use the common _.fit()_ and
_.transform()_ functions (sklearn-like interface)

```python
import dcp
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

model = dcp.ComplexityProfile()
model.fit(X, y)
model.transform()
```

Complexity profile takes different inputs from none to
specific measures to be obtained.

## References

[1] How Complex is your classification problem? A survey on measuring
classification complexity, [ArXiv](https://arxiv.org/abs/1808.03591)

[2] The Extended Complexity Library (ECoL),
[github repo](https://github.com/lpfgarcia/ECoL)
