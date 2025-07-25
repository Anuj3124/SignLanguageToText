# utils/sentence_builder.py
class SentenceBuilder:
    def __init__(self):
        self.words = []

    def add_word(self, word):
        if not self.words or word != self.words[-1]:
            self.words.append(word)

    def get_sentence(self):
        return ' '.join(self.words)

    def reset(self):
        self.words = []
