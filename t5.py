import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, BatchNormalization, Conv1D, MaxPooling1D, Dropout, LayerNormalization
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import gensim
from gensim.models import Word2Vec
from indicnlp.tokenize import indic_tokenize
import fasttext
import fasttext.util
import re
import string
import gc
import os

# Import transformers for mT5 paraphrasing
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class HindiFakeNewsDetector:
    def __init__(self, max_features=10000, max_len=100):  
        self.max_features = max_features
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_features)
        self.embeddings_dim = 300
        
        # Initialize mT5 paraphrasing model
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
    
    def process_labels(self, labels_set):
        labels = labels_set.fillna('')
        fake_keywords = ['fake']
        def check_fake_labels(label_string):
            labels = label_string.lower().split(',')
            return any(any(keyword in label for keyword in fake_keywords) for label in labels)
    
        return labels.apply(check_fake_labels).astype(int)

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        tokens = indic_tokenize.trivial_tokenize(text)
        return ' '.join(tokens)

    def paraphrase_text(self, text, num_paraphrases=4):
        """
        Generate paraphrases using mT5 model
        
        Args:
            text (str): Input text to paraphrase
            num_paraphrases (int): Number of paraphrases to generate
        
        Returns:
            list: List of paraphrased texts
        """
        try:
            # Tokenize the input text
            inputs = self.paraphrase_tokenizer(
                text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # Generate multiple paraphrases
            paraphrases = []
            for _ in range(num_paraphrases):
                outputs = self.paraphrase_model.generate(
                    **inputs, 
                    max_length=512, 
                    num_return_sequences=1, 
                    do_sample=True, 
                    top_p=0.9, 
                    temperature=0.7
                )
                
                # Decode the generated text
                paraphrased_text = self.paraphrase_tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
                paraphrases.append(paraphrased_text)
            
            return paraphrases
        except Exception as e:
            print(f"Paraphrasing error: {e}")
            return [text]  # Return original text if paraphrasing fails

    def balance_dataset(self, texts, labels):
        """
        Balance the dataset using mT5 paraphrasing
        
        Args:
            texts (pd.Series): Original texts
            labels (pd.Series): Original labels
        
        Returns:
            tuple: Balanced texts and labels
        """
        # Separate fake and non-fake samples
        fake_indices = np.where(labels == 1)[0]
        non_fake_indices = np.where(labels == 0)[0]
        
        # Get original counts
        fake_count = len(fake_indices)
        non_fake_count = len(non_fake_indices)
        
        # Determine the target count (majority class)
        target_count = max(fake_count, non_fake_count)
        
        # Initialize lists to store balanced data
        balanced_texts = list(texts)
        balanced_labels = list(labels)
        
        # If fake samples are minority, generate paraphrases
        if fake_count < non_fake_count:
            for idx in fake_indices:
                # Generate paraphrases for fake samples
                paraphrases = self.paraphrase_text(texts.iloc[idx])
                for paraphrase in paraphrases:
                    balanced_texts.append(paraphrase)
                    balanced_labels.append(1)  # Keep the label as fake
        
        # If non-fake samples are minority, generate paraphrases
        elif non_fake_count < fake_count:
            for idx in non_fake_indices:
                # Generate paraphrases for non-fake samples
                paraphrases = self.paraphrase_text(texts.iloc[idx])
                for paraphrase in paraphrases:
                    balanced_texts.append(paraphrase)
                    balanced_labels.append(0)  # Keep the label as non-fake
        
        print(f"Original dataset shape: {np.bincount(labels)}")
        print(f"Balanced dataset shape: {np.bincount(balanced_labels)}")
        
        # Tokenize and pad the balanced texts
        self.tokenizer.fit_on_texts(balanced_texts)
        sequences = self.tokenizer.texts_to_sequences(balanced_texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = np.array(balanced_labels)
        
        return X, y

    def prepare_data(self, df, balance=True):
        texts = df['Post'].apply(self.preprocess_text)
        labels = self.process_labels(df['Labels Set'])

        print("\nDataset Info (Before Balancing):")
        print(f"Fake posts: {sum(labels == 1)}")
        print(f"Non-Fake posts: {sum(labels == 0)}")

        if balance:
            X, y = self.balance_dataset(texts, labels)
            print("\nDataset Info (After Balancing):")
            print(f"Fake posts: {sum(y == 1)}")
            print(f"Non-Fake posts: {sum(y == 0)}")
        else:
            print("Tokenizing texts...")
            self.tokenizer.fit_on_texts(texts)
            sequences = self.tokenizer.texts_to_sequences(texts)
            X = pad_sequences(sequences, maxlen=self.max_len)
            y = labels.values

        return X, y
    

    def load_word2vec_embeddings(self, texts, chunk_size=1000):
        print("Training Word2Vec model...")
        # Process texts in chunks to save memory
        tokenized_texts = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size].apply(lambda x: str(x).split())
            tokenized_texts.extend(chunk.values)
            gc.collect()  # Force garbage collection

        word2vec_model = Word2Vec(sentences=tokenized_texts,
                                vector_size=self.embeddings_dim,
                                window=5,
                                min_count=1,
                                workers=4)

        print("Creating embedding matrix...")
        embedding_matrix = np.zeros((self.max_features, self.embeddings_dim))
        for word, idx in self.tokenizer.word_index.items():
            if idx < self.max_features:
                try:
                    embedding_matrix[idx] = word2vec_model.wv[word]
                except KeyError:
                    continue

        del word2vec_model
        gc.collect()
        return embedding_matrix

    def load_fasttext_embeddings(self, model_path):
        print(f"Loading FastText model from {model_path}...")
        try:
            # Load model with memory-mapping
            #fasttext.util.download_model('hi', if_exists='ignore')
            model = fasttext.load_model(model_path)

            print("Creating embedding matrix...")
            embedding_matrix = np.zeros((self.max_features, self.embeddings_dim))
            for word, idx in self.tokenizer.word_index.items():
                if idx < self.max_features:
                    embedding_matrix[idx] = model.get_word_vector(word)[:self.embeddings_dim]

            del model
            gc.collect()
            return embedding_matrix
        except Exception as e:
            print(f"Error loading FastText model: {e}")
            print("Returning zero embedding matrix instead")
            return np.zeros((self.max_features, self.embeddings_dim))

    def create_cnn_bilstm_model(self, embedding_matrix):
        model = Sequential([
            Embedding(self.max_features, embedding_matrix.shape[1],
                     weights=[embedding_matrix], trainable=False),
            Conv1D(128, 7, activation='relu'),
            MaxPooling1D(2),
            BatchNormalization(),
            Dropout(0.5),
            
            # Bidirectional LSTM layers with increased units
            Bidirectional(LSTM(128, return_sequences=True)),
            LayerNormalization(),
            Dropout(0.5),

            Bidirectional(LSTM(64)),
            LayerNormalization(),
            Dropout(0.5),
            
            Dense(128, activation='relu'),
            LayerNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def create_lstm_model(self, embedding_matrix):
        model = Sequential([
            Embedding(self.max_features, embedding_matrix.shape[1],
                     weights=[embedding_matrix], trainable=False),
            LSTM(64, return_sequences=True),  # Reduced units
            LSTM(32),  # Reduced units
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def create_bilstm_model(self, embedding_matrix):
        model = Sequential([
            Embedding(self.max_features, embedding_matrix.shape[1],
                     weights=[embedding_matrix], trainable=False),
            Bidirectional(LSTM(64, return_sequences=True)),  # Reduced units
            Bidirectional(LSTM(32)),  # Reduced units
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def create_cnn_lstm_model(self, embedding_matrix):
        model = Sequential([
            Embedding(self.max_features, embedding_matrix.shape[1],
                     weights=[embedding_matrix], trainable=False),
            Conv1D(64, 5, activation='relu'),  # Reduced filters
            MaxPooling1D(2),
            LSTM(64, return_sequences=True),  # Reduced units
            LSTM(32),  # Reduced units
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

def main():
    print("Loading datasets...")
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('valid.csv')
    test_df = pd.read_csv('test.csv')

    print("\nDataset Info:")
    print(train_df['Labels Set'].value_counts())

    # Initialize detector with reduced parameters
    detector = HindiFakeNewsDetector()

    print("\nPreparing data...")
    X_train, y_train = detector.prepare_data(train_df, balance=True)
    X_val, y_val = detector.prepare_data(val_df, balance=False)
    X_test, y_test = detector.prepare_data(test_df, balance=False)

    # Define which embeddings to use
    embeddings = {
        'indicnlp': detector.load_fasttext_embeddings('indicnlp.v1.hi.bin')
    }

    model_creators = {
        'cnn_bilstm': detector.create_cnn_bilstm_model,
    }

    results = {}
    for embed_name, embedding_matrix in embeddings.items():
        for model_name, model_creator in model_creators.items():
            print(f"\nTraining {model_name} with {embed_name} embeddings")

            model = model_creator(embedding_matrix)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=32,
                verbose=1
            )

            test_loss, test_accuracy = model.evaluate(X_test, y_test)
            y_pred = (model.predict(X_test) > 0.5).astype(int)

            results[f"{embed_name}_{model_name}"] = {
                'test_accuracy': test_accuracy,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

            # Save model and predictions
            model.save(f"model_{embed_name}_{model_name}.h5")
            predictions_df = pd.DataFrame({
                'Unique ID': test_df['Unique ID'],
                'Actual Label': y_test,
                'Predicted Label': y_pred.flatten(),
                'Original Labels Set': test_df['Labels Set']
            })
            predictions_df.to_csv(f"predictions_{embed_name}_{model_name}.csv", index=False)

            # Clear memory
            del model
            gc.collect()

    for model_name, result in results.items():
        print(f"\nResults for {model_name}:")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        print("\nClassification Report:")
        print(result['classification_report'])
        print("\nConfusion Matrix:")
        print(result['confusion_matrix'])

if __name__ == "__main__":
    main()