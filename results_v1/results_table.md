| Approach | Trainable params | Time (s) | Train acc | Val acc | Test acc | Test macro-F1 |
|---|---|---|---|---|---|---|
| full | 1,555,781 | 736.2 | 0.9878 | 0.8750 | 0.8302 | 0.8248 |
| head | 628,773 | 213.7 | 0.9840 | 0.8791 | 0.8585 | 0.8581 |
| features_logreg | 0 | 36.9 | 0.9881 | 0.8764 | 0.8569 | 0.8551 |
| features_linear_svm | 0 | 42.8 | 0.9990 | 0.8750 | 0.8340 | 0.8314 |
| features_random_forest | 0 | 38.1 | 1.0000 | 0.8682 | 0.8256 | 0.8203 |