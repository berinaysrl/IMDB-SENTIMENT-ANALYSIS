import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from data_processing_visualizations import (
    load_and_preprocess_data,
    plot_wordclouds,
    plot_training_history,
    plot_training_loss,
    plot_confusion_matrix_func
)
from model import build_model, train_and_evaluate_model

if __name__ == "__main__":
    csv_path = "/Users/berinayzumrasariel/Desktop/imdb sentiment project/data/IMDB Dataset.csv"

    data, train_padded, test_padded, y_train, y_test, tokenizer = load_and_preprocess_data(csv_path)

    plot_wordclouds(data)

    model = build_model(max_words=10000, embedding_dim=16, max_len=200)
    model.summary()

    history = train_and_evaluate_model(model, train_padded, y_train, test_padded, y_test, epochs=10)

    loss, accuracy = model.evaluate(test_padded, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    y_pred_prob = model.predict(test_padded)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    plot_training_history(history)
    plot_training_loss(history)
    plot_confusion_matrix_func(y_test, y_pred)

    plt.show()
