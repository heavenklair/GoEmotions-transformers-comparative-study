import tensorflow as tf
from transformers import TFDistilBertModel

class DistilBertClassificationModel:

    def __init__(self, checkpoint, max_length, learning_rate=0.00005, dropout=0.1, num_classes=7):
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int64, name='input_ids_layer')
        attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int64, name='attention_mask_layer')

        distilbert = TFDistilBertModel.from_pretrained(self.checkpoint)
        distilbert.trainable = True

        distilbert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        distilbert_out = distilbert(distilbert_inputs, output_hidden_states=True, output_attentions=True)
        
        # Getting the last hidden state, DistilBERT doesn't have a pooler like BERT
        last_hidden_state = distilbert_out['last_hidden_state']
        cls_token = last_hidden_state[:, 0, :]
        
        hidden = tf.keras.layers.Dropout(self.dropout)(cls_token)
        classification_logits = tf.keras.layers.Dense(self.num_classes, name='classification_layer')(hidden)

        classification_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=classification_logits)
        classification_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

        return classification_model
