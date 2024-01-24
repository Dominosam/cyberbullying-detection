import spacy
import re
from spacy.lang.pl.stop_words import STOP_WORDS

class TextPreprocessor:
    def __init__(self, language_model="pl_core_news_sm"):
        try:
            self.nlp = spacy.load(language_model)
        except IOError:
            spacy.cli.download(language_model)
            self.nlp = spacy.load(language_model)

    def preprocess(self, text):
        text = text.lower()
        tokens = [token.lemma_ for token in self.nlp(text)]
        processed_text = " ".join(tokens)

        # Replacing mentions and URLs
        processed_text = re.sub(r'@\w+', 'USER_MENTION', processed_text)
        processed_text = re.sub(r'https?://\S+', 'URL', processed_text)

        # Removing special characters and filtering stopwords
        processed_text = re.sub(r'[^\w\s]', '', processed_text)
        processed_text = " ".join([word for word in processed_text.split() if word not in STOP_WORDS])

        return processed_text
