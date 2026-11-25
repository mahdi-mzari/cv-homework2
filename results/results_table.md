| Approach | Trainable params | Time (s) | Train acc | Val acc | Test acc | Test macro-F1 |
|---|---|---|---|---|---|---|
| full | 1,555,781 | 554.5 | 0.9915 | 0.8845 | 0.8455 | 0.8419 |
| head | 628,773 | 593.5 | 0.9990 | 0.8981 | 0.8692 | 0.8670 |
| features_logreg | 0 | 47.1 | 0.9881 | 0.8764 | 0.8569 | 0.8551 |
| features_linear_svm | 0 | 60.0 | 0.9990 | 0.8750 | 0.8340 | 0.8314 |
| features_random_forest | 0 | 52.8 | 1.0000 | 0.8682 | 0.8256 | 0.8203 |