from models.text_cnn import TextCnn
from models.text_rnn import TextRnn
from data_processor import *
import time


class LabTextClassification:
    def __init__(self, model_instance):
        self.model = model_instance

    def train(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        t1 = time.time()
        X_train, y_train = load_text_data(lines_train)

        print("数据耗时", time.time() - t1)
        self.model.fit(X_train, y_train, batch_size=128, epochs=5)

    def predict(self):
        X_test, y_test = load_text_data(lines_test)
        score = self.model.evaluate(X_test, y_test)
        print('acc', score[1])

        print("开始预测：-------------------------------------")

        print(self.model.predict(X_test))
        for res in self.model.predict(X_test[:10]):
            print(res.argmax())


if __name__ == "__main__":
    text_cnn_model = TextCnn(
        vocab_size=20000,
        embedding_dim=100,
        num_filters=128,
        num_classes=14,
        sequence_length=600
    )
    text_rnn_model = TextRnn(
        vocab_size=20000,
        embedding_dim=128,
        hidden_size=100,
        num_classes=14,
        sequence_length=600
    )
    model = LabTextClassification(text_rnn_model)
    model.train()
    model.predict()
