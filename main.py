from src.data_preprocessing import load_data, preprocess_data, split_data
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    # Load data
    data = load_data("data/winequality-white.csv")

    # Preprocess
    X, y = preprocess_data(data)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
