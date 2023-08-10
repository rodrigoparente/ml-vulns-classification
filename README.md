# A Vulnerability Risk Assessment Methodology Using Active Learning

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

For more information about the machine learning techniques tested and the vulnerability risk classification problem, read the paper: [A Vulnerability Risk Assessment Methodology Using Active Learning](https://link.springer.com/chapter/10.1007/978-3-031-28451-9_15), published at the international conference on Advanced Information Networking and Applications. Also, with you would like to know more about the security dataset used in the experiments, you can check the [CVEJoin Security Dataset](https://github.com/rodrigoparente/cvejoin-security-dataset) repository.

## Requirements

Install requirements using the following command

```bash
$ pip install -r requirements.txt
```

## Execute

Execute the code using the following command

```bash
$ python main.py
```

## Output

The results will be placed at the ``results/compiled`` folder.

## Optional Configuration

The user can configure two optional environment variables. The first, `CHAT_ID`, is the identifier of the shared chat to which Telegram's bot will post the status message of the experiment. The second, `BOT_ID`, is the unique identifier of Telegram's bot. For more information about it, you can read the [Telegram Bot](https://core.telegram.org/bots/api) API.

## Reference 

If you re-use this work, please cite:

```
@inproceedings{da2023vulnerability,
  title={A Vulnerability Risk Assessment Methodology Using Active Learning},
  author={da Ponte, Francisco RP and Rodrigues, Emanuel B and Mattos, C{\'e}sar LC},
  booktitle={International Conference on Advanced Information Networking and Applications},
  pages={171--182},
  year={2023},
  organization={Springer}
}
```

## License

This project is [GNU GPLv3 licensed](./LICENSE).