import random

from spacy.lang.pl import Polish
from spacy.pipeline.textcat import Config, single_label_cnn_config
from sklearn.metrics import precision_score, recall_score, f1_score
from spacy.training import Example

def compounding(start, stop, compound):
    yield start
    while start < stop:
        start *= compound
        yield start


class TextCategorization:
    def __init__(self):
        self.nlp = Polish()
        self.textcat = self.setup_textcat()

    def setup_textcat(self):
        config = Config().from_str(single_label_cnn_config)
        textcat = self.nlp.add_pipe("textcat", config=config)
        textcat.add_label("cyberbullying")
        textcat.add_label("non-cyberbullying")
        return textcat

    def train(self, train_texts, train_tags, n_iter=10):
        train_data = list(zip(train_texts, [{"cats": {"cyberbullying": label == 1, "non-cyberbullying": label == 0}} for label in train_tags]))
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "textcat"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            for i in range(n_iter):
                random.shuffle(train_data)
                losses = {}
                for batch in self.minibatch(train_data):
                    examples = [Example.from_dict(self.nlp.make_doc(text), annotations) for text, annotations in batch]
                    self.nlp.update(examples, drop=0.5, losses=losses)

    def evaluate(self, test_texts, test_tags):
        test_data = list(zip(test_texts, [{"cats": {"cyberbullying": label == 1, "non-cyberbullying": label == 0}} for label in test_tags]))
        precision, recall, f1_score = self.evaluate_model(test_data)
        return precision, recall, f1_score

    def evaluate_model(self, eval_data):
        preds = []
        true_labels = []
        for text, annotation in eval_data:
            doc = self.nlp(text)
            true_label = annotation["cats"]["cyberbullying"]
            predicted_label = doc.cats["cyberbullying"] > 0.5
            preds.append(predicted_label)
            true_labels.append(true_label)

        precision = precision_score(true_labels, preds)
        recall = recall_score(true_labels, preds)
        f1 = f1_score(true_labels, preds)

        return precision, recall, f1

    def minibatch(self, data):
        length = len(data)
        if length == 0:
            return []
        size_gen = compounding(4.0, 32.0, 1.001)
        for i in range(0, length, int(next(size_gen))):
            yield data[i:min(i + int(next(size_gen)), length)]
