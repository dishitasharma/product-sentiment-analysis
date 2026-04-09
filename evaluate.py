from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

results = {}

def evaluate_save(model, name):
    preds = model.predict(X_test)

    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average='weighted'),
        "recall": recall_score(y_test, preds, average='weighted'),
        "f1_score": f1_score(y_test, preds, average='weighted')
    }

evaluate_save(svm, "SVM")
evaluate_save(rf, "Random Forest")
evaluate_save(xgb, "XGBoost")
