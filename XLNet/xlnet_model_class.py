import tensorflow as tf
from transformers import TFXLNetModel

class XLNETClassificationModel:

    def __init__(self, checkpoint, max_length, learning_rate=0.00005, dropout=0.1, num_classes=28):
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        """
        Builds and returns the xlnet classification model.
        """
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int64, name='input_ids_layer')
        token_type_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int64, name='token_type_ids_layer')
        attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int64, name='attention_mask_layer')

        xlnet = TFXLNetModel.from_pretrained(self.checkpoint)
        xlnet.trainable = True

        xlnet_inputs =  { 'input_ids': input_ids,
                          'token_type_ids': token_type_ids,
                          'attention_mask': attention_mask}

        xlnet_out = xlnet(xlnet_inputs, output_hidden_states=True, output_attentions=True) # returns additional outputs 
        
        cls_token_representation = xlnet_out[0][:, 0, :]
        hidden_states = xlnet_out['hidden_states']
        attentions = xlnet_out['attentions']
    
        hidden = tf.keras.layers.Dropout(self.dropout)(cls_token_representation) # dropout layer
        
        
        classification_logits = tf.keras.layers.Dense(self.num_classes, name='classification_layer')(hidden)
        outputs = [classification_logits, hidden_states, attentions] 

        classification_model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], 
                                              outputs=classification_logits)        
        
        classification_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

        return classification_model
