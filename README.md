# Polisis Classifiers

This repository builds on the following code https://github.com/SmartDataAnalytics/Polisis_Benchmark to reproduce the results for privacy policy classification from Harkous et al. (2018).
It also contains the trained classifiers that can be used as they are, as well as domain-specific word-embeddings.

* Use the file `CNN Multilabel Classifier.ipynb` to train the classifier for the main category.
* Use the file `CNN Multilabel Classifier Attributes.ipynb` to train the classifiers for the attributes.
* Use the file `predict.py` to get predictions for a privacy policies. `predict.py` expects the privacy policy to be predicted as an array of strings.


Harkous, H., Fawaz, K., Lebret, R., Schaub, F., Shin, K. G., & Aberer, K. (2018). Poli- sis: Automated analysis and presentation of privacy policies using deep learning. In Proceedings of the 27th usenix conference on security symposium (p. 531–548). USA: USENIX Association.
