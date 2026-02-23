from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, selector, X_test, y_test):
    """
    Evaluate trained model.
    """

    # Transform test set using same selector
    X_test_selected = selector.transform(X_test)

    y_pred = model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))