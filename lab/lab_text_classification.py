from models.text_cnn import TextCNN
from models.text_rnn import TextRNN
from models.text_rcnn import TextRCNN
from models.han import HAN
from models.dpcnn import DPCNN
from data_process.data_processor import *
import time
import datetime


class LabTextClassification:
    def __init__(self, model_instance):
        self.model = model_instance
        if type(self.model) == HAN:
            self.han_flag = True
        else:
            self.han_flag = False
        self.max_len = 600

    def train(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        t1 = time.time()
        X_train, y_train = load_text_data(lines_train, max_len=self.max_len, han=self.han_flag)

        print("数据耗时", time.time() - t1)
        x_test, y_test = load_text_data(lines_val, max_len=self.max_len, han=self.han_flag)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.fit(X_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test),
                       callbacks=[tensorboard_callback])

        print('export saved models.')

    def predict(self):
        t1 = time.time()
        X_test, y_test = load_text_data(lines_test, max_len=self.max_len, han=self.han_flag)
        print("数据耗时", time.time() - t1)
        score = self.model.evaluate(X_test, y_test)
        print('acc', score[1])

        print("开始预测：-------------------------------------")

        print(self.model.predict(X_test))
        for res in self.model.predict(X_test[:10]):
            print(res.argmax())


if __name__ == "__main__":
    text_cnn_model = TextCNN(
        vocab_size=20000,
        embedding_dim=100,
        num_filters=128,
        num_classes=14,
        sequence_length=600
    )
    text_rnn_model = TextRNN(
        vocab_size=20000,
        embedding_dim=128,
        hidden_size=100,
        num_classes=14,
        sequence_length=600
    )
    text_rcnn_model = TextRCNN(
        vocab_size=20000,
        embedding_dim=128,
        rnn_hidden_size=100,
        num_filters=150,
        num_classes=14,
        sequence_length=600
    )
    han_model = HAN(
        vocab_size=20000,
        embedding_dim=128,
        num_classes=14,
        sentence_length=30,
        doc_length=20,
        hidden_size=100
    )
    dpcnn_model = DPCNN(
        vocab_size=20000,
        embedding_dim=300,
        num_classes=14,
        sentence_length=600,
        hidden_size=256,
        num_filters=256
    )
    model = LabTextClassification(dpcnn_model)
    model.train()
    model.predict()
