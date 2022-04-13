# Vulnerability Risk Classification using ML

The code from this repository is responsible to test four different machine learning techniques in the vulnerability risk classification problem. The techniques tested are:

 - Active-learning using a supervised algorithm 
 - Active-learning using a semi-supervised algorithm
 - Random labelling using a supervised algorithm
 - Random labelling using a semi-supervised algorithm

The machine learning algorithms used in the test are:

 - [Random Forest](https://en.wikipedia.org/wiki/Random_forest)
 - [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
 - [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
 - [Support Vector Machine](https://en.wikipedia.org/wiki/Support-vector_machine)
 - [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)

More information on the dataset used in the problem can be found [here](https://github.com/rodrigoparente/vulns-data-agg).

# Requirements

Install requirements using the following command

```bash
$ pip install -r requirements.txt
```

# Execute

Execute the code using the following command

```bash
$ python main.py
```

# Output

The results will be placed at the ``results/compiled`` folder.

# License

This project is [GNU GPLv3 licensed](./LICENSE).