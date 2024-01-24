import os
from sklearn.model_selection import train_test_split
from text_preprocessor import TextPreprocessor
from text_categorization import TextCategorization
from text_cnn_classifier import TextCNNClassifier
import matplotlib.pyplot as plt

TRAIN_DATA_DIR = "resources/train"
TEST_SIZE = 0.6
RANDOM_STATE = 42
MAX_SEQUENCE_LENGTH = 110
VOCAB_SIZE = 13000
EMBEDDING_DIM = 100
NUM_CLASSES = 2

class CyberbullyingDetection:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.text_classifier = TextCategorization()
        self.text_cnn_classifier = TextCNNClassifier(MAX_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES)

    def gather_train_data(self):
        train_texts, train_tags = [], []
        text_file = os.path.join(TRAIN_DATA_DIR, "test_text.txt")
        tags_file = os.path.join(TRAIN_DATA_DIR, "test_tags.txt")

        with open(text_file, 'r', encoding='utf-8') as file, open(tags_file, 'r', encoding='utf-8') as tag_file:
            train_texts = [line.strip() for line in file]
            train_tags = [int(line.strip()) for line in tag_file]

        return train_texts, train_tags

    def plot_metrics(self, metrics_tc, metrics_cnn):
        labels = ['Precision', 'Recall', 'F1 Score']
        textcat_metrics = [metrics_tc['precision'], metrics_tc['recall'], metrics_tc['f1_score']]
        cnn_metrics = [metrics_cnn['precision'], metrics_cnn['recall'], metrics_cnn['f1_score']]

        x = range(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x, textcat_metrics, width, label='TextCategorization')
        rects2 = ax.bar([p + width for p in x], cnn_metrics, width, label='TextCNNClassifier')

        ax.set_ylabel('Scores')
        ax.set_title('Scores by model and metric')
        ax.set_xticks([p + width / 2 for p in x])
        ax.set_xticklabels(labels)
        ax.legend()

        self.add_value_labels(ax, rects1)
        self.add_value_labels(ax, rects2)

        plt.show()

    def add_value_labels(self, ax, rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    def train_and_evaluate(self):
        train_texts, train_tags = self.gather_train_data()
        train_texts = [self.preprocessor.preprocess(text) for text in train_texts]
        X_train, X_test, y_train, y_test = train_test_split(train_texts, train_tags, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        self.text_classifier.train(X_train, y_train)
        precision_tc, recall_tc, f1_score_tc = self.text_classifier.evaluate(X_test, y_test)

        self.text_cnn_classifier.train(X_train, y_train, epochs=40, batch_size=20)
        precision_cnn, recall_cnn, f1_score_cnn = self.text_cnn_classifier.evaluate(X_test, y_test)

        self.display_metrics("TextCategorizer", precision_tc, recall_tc, f1_score_tc)
        self.display_metrics("TextCNNClassifier", precision_cnn, recall_cnn, f1_score_cnn)

        metrics_tc = {'precision': precision_tc, 'recall': recall_tc, 'f1_score': f1_score_tc}
        metrics_cnn = {'precision': precision_cnn, 'recall': recall_cnn, 'f1_score': f1_score_cnn}

        self.plot_metrics(metrics_tc, metrics_cnn)

    def display_metrics(self, model_name, precision, recall, f1_score):
        print(f"{model_name} Metrics:")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

if __name__ == "__main__":
    detector = CyberbullyingDetection()
    detector.train_and_evaluate()