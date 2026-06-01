import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, Dense, Dropout, TimeDistributed,
    GlobalMaxPooling1D, Concatenate, RNN
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tensorflow.keras.optimizers import Adam

# ========= USER CONFIG ========= #
data_size   = None
max_epochs  = 40
batch_size  = 128
random_state = 42

excel_path = "/kaggle/input/fulltag/padt_pos_4cols_fulltags.xlsx"
pkl_path   = "/kaggle/working/preprocessed_pos_data_padt_word_char_lstm_sde_ar1.pkl"

MAX_LEN        = 100
MAX_WORD_LEN   = 15
WORD_EMB_DIM   = 128
CHAR_EMB_DIM   = 50
LSTM_UNITS     = 128
NOISE_STD      = 0.5   # std للضوضاء الابتدائية ε_t
AR_RHO         = 0.8   # معامل AR(1)
# =============================== #

# ========== DATA PREPROCESS (CACHED) ========== #
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    X_words = data["X_words"]
    X_chars = data["X_chars"]
    y = data["y"]
    sentences = data["sentences"]
    tag_tokenizer = data["tag_tokenizer"]
    word_tokenizer = data["word_tokenizer"]
    char_tokenizer = data["char_tokenizer"]
    print("Loaded preprocessed WORD+CHAR data from pickle.")
else:
    df = pd.read_excel(excel_path)

    sent_cols = ["train_sentence", "dev_sentence", "test_sentence"]
    pos_cols  = ["train_pos_seq",  "dev_pos_seq",  "test_pos_seq"]

    all_sentences, all_tags_seq = [], []
    for s_col, p_col in zip(sent_cols, pos_cols):
        s = df[s_col].fillna("").astype(str).values
        p = df[p_col].fillna("").astype(str).values
        for sent, tags in zip(s, p):
            if sent.strip() == "" or tags.strip() == "":
                continue
            all_sentences.append(sent)
            all_tags_seq.append(tags)

    sentences = np.array(all_sentences)
    tags_seq  = np.array(all_tags_seq)

    if data_size is not None and data_size <= len(sentences):
        sentences = sentences[:data_size]
        tags_seq  = tags_seq[:data_size]

    print("Total sentences:", len(sentences))

    # WORD tokenizer
    word_tokenizer = Tokenizer(char_level=False, oov_token="[OOV]")
    word_tokenizer.fit_on_texts(sentences)
    word_seqs = word_tokenizer.texts_to_sequences(sentences)
    X_words = pad_sequences(word_seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    # CHAR tokenizer
    all_words_for_chars = []
    for sent in sentences:
        all_words_for_chars.extend(sent.split())

    char_tokenizer = Tokenizer(char_level=True, oov_token="[OOV_CH]")
    char_tokenizer.fit_on_texts(all_words_for_chars)

    def sentence_to_char_ids(sent):
        words = sent.split()
        words = words[:MAX_LEN] + [""] * max(0, MAX_LEN - len(words))
        char_ids = []
        for w in words:
            chars = list(w)
            ch_seq = char_tokenizer.texts_to_sequences(["".join(chars)])[0]
            ch_seq = ch_seq[:MAX_WORD_LEN]
            ch_seq = ch_seq + [0] * max(0, MAX_WORD_LEN - len(ch_seq))
            char_ids.append(ch_seq)
        return np.array(char_ids, dtype="int32")

    X_chars = np.stack([sentence_to_char_ids(s) for s in sentences])

    # Tag tokenizer
    tag_tokenizer = Tokenizer()
    tag_tokenizer.fit_on_texts(tags_seq)
    y_seqs = tag_tokenizer.texts_to_sequences(tags_seq)
    y = pad_sequences(y_seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    with open(pkl_path, "wb") as f:
        pickle.dump({
            "X_words": X_words,
            "X_chars": X_chars,
            "y": y,
            "sentences": sentences,
            "tag_tokenizer": tag_tokenizer,
            "word_tokenizer": word_tokenizer,
            "char_tokenizer": char_tokenizer
        }, f)
    print("Saved preprocessed WORD+CHAR data to pickle.")

# ====== SHAPES ====== #
vocab_size_words = len(word_tokenizer.word_index) + 1
vocab_size_chars = len(char_tokenizer.word_index) + 1
tag_count        = len(tag_tokenizer.word_index) + 1
max_len          = X_words.shape[1]
max_word_len     = X_chars.shape[2]

print("vocab_size_words:", vocab_size_words)
print("vocab_size_chars:", vocab_size_chars)
print("tag_count       :", tag_count)
print("max_len(words)  :", max_len)
print("max_word_len    :", max_word_len)

# ====== SPLIT ====== #
Xw_trainval, Xw_test, Xc_trainval, Xc_test, y_trainval, y_test = train_test_split(
    X_words, X_chars, y, test_size=0.1, random_state=random_state
)
Xw_train, Xw_val, Xc_train, Xc_val, y_train, y_val = train_test_split(
    Xw_trainval, Xc_trainval, y_trainval, test_size=0.1111, random_state=random_state
)
print("Train words:", Xw_train.shape, "chars:", Xc_train.shape, "y:", y_train.shape)
print("Val   words:", Xw_val.shape, "chars:", Xc_val.shape, "y:", y_val.shape)
print("Test  words:", Xw_test.shape, "chars:", Xc_test.shape, "y:", y_test.shape)

# ====== LSTM-SDE CELL WITH AR(1) NOISE ON h_t ====== #
class LSTMSDECellH_AR1(tf.keras.layers.Layer):
    def __init__(self, units, noise_std=1.0, ar_rho=0.8, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # 3 حالات: h, c, z (ضوضاء AR(1) السابقة)
        self.state_size = [units, units, units]
        self.output_size = units
        self.noise_std = noise_std
        self.ar_rho = ar_rho

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.W_x = self.add_weight(
            shape=(input_dim, 4 * self.units),
            initializer="glorot_uniform",
            name="W_x_h_ar1",
        )
        self.W_h = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer="orthogonal",
            name="W_h_h_ar1",
        )
        self.b = self.add_weight(
            shape=(4 * self.units,),
            initializer="zeros",
            name="b_h_ar1",
        )

        self.drift_dense_h = Dense(self.units, name="drift_dense_h_ar1")
        self.diff_dense_h  = Dense(self.units, name="diff_dense_h_ar1")

        super().build(input_shape)

    def call(self, inputs, states, training=None):
        h_tm1, c_tm1, z_tm1 = states

        # LSTM القياسي
        z = tf.matmul(inputs, self.W_x) + tf.matmul(h_tm1, self.W_h) + self.b
        zi, zf, zc, zo = tf.split(z, num_or_size_splits=4, axis=-1)

        i = tf.sigmoid(zi)
        f = tf.sigmoid(zf)
        o = tf.sigmoid(zo)
        c_hat = tf.tanh(zc)

        c_t = f * c_tm1 + i * c_hat
        h_t = o * tf.tanh(c_t)

        # ضوضاء AR(1) على h_t
        eps_t = tf.random.normal(tf.shape(h_t), mean=0.0, stddev=self.noise_std)
        z_t = self.ar_rho * z_tm1 + eps_t

        dt = 1.0
        drift_h = self.drift_dense_h(h_t)
        diffusion_h = self.diff_dense_h(h_t)

        h_t_sde = h_t + drift_h * dt + diffusion_h * z_t * tf.sqrt(dt)

        return h_t_sde, [h_t_sde, c_t, z_t]

# ====== MODEL: word+char -> BiLSTM-SDE(AR1 on h_t) -> TimeDistributed ====== #
word_inputs = Input(shape=(max_len,), name="word_input")
word_emb = Embedding(vocab_size_words, WORD_EMB_DIM,
                     mask_zero=False, name="word_embedding")(word_inputs)

char_inputs = Input(shape=(max_len, max_word_len), name="char_input")
char_emb_layer = Embedding(vocab_size_chars, CHAR_EMB_DIM,
                           mask_zero=False, name="char_embedding")
char_emb = TimeDistributed(char_emb_layer, name="td_char_emb")(char_inputs)

char_cnn = TimeDistributed(
    Conv1D(filters=50, kernel_size=3, padding="same", activation="relu"),
    name="td_char_cnn"
)(char_emb)
char_pool = TimeDistributed(GlobalMaxPooling1D(), name="td_char_pool")(char_cnn)

merged = Concatenate(axis=-1, name="word_char_concat")([word_emb, char_pool])

cell_f = LSTMSDECellH_AR1(LSTM_UNITS, noise_std=NOISE_STD, ar_rho=AR_RHO)
cell_b = LSTMSDECellH_AR1(LSTM_UNITS, noise_std=NOISE_STD, ar_rho=AR_RHO)

lstm_sde_f = RNN(cell_f,
                 return_sequences=True,
                 go_backwards=False,
                 name="lstm_sde_f_ar1")(merged)

lstm_sde_b = RNN(cell_b,
                 return_sequences=True,
                 go_backwards=True,
                 name="lstm_sde_b_ar1")(merged)

bilstm_sde = Concatenate(axis=-1, name="bilstm_sde_ar1")([lstm_sde_f, lstm_sde_b])

logits = TimeDistributed(Dense(tag_count, activation="softmax"),
                         name="tag_output")(bilstm_sde)

model = Model(inputs=[word_inputs, char_inputs], outputs=logits)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ====== TRAIN (بدون EarlyStopping) ====== #
y_train_targets = np.expand_dims(y_train, -1)
y_val_targets   = np.expand_dims(y_val, -1)

checkpoint = ModelCheckpoint(
    "/kaggle/working/best_word_char_bilstm_sde_ar1_pos.keras",
    monitor="val_loss",
    save_best_only=True
)

history = model.fit(
    [Xw_train, Xc_train], y_train_targets,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=([Xw_val, Xc_val], y_val_targets),
    callbacks=[checkpoint]
)

# ====== EVALUATION ====== #
y_probs = model.predict([Xw_test, Xc_test])
y_pred = np.argmax(y_probs, axis=-1)
y_true = y_test

y_true = np.squeeze(y_true)
y_pred = np.squeeze(y_pred)

mask = (y_true != 0)
y_true_flat = y_true[mask]
y_pred_flat = y_pred[mask]

idx2tag = {v: k for k, v in tag_tokenizer.word_index.items()}
idx2tag[0] = '[PAD]'
y_true_tags = [idx2tag[idx] for idx in y_true_flat]
y_pred_tags = [idx2tag[idx] for idx in y_pred_flat]

print("\nTest Classification Report:")
with open("/kaggle/working/test_classification_report_word_char_bilstm_sde_ar1.txt",
          "w", encoding="utf-8") as f:
    f.write(classification_report(y_true_tags, y_pred_tags))
print(classification_report(y_true_tags, y_pred_tags))

labels = np.unique(y_true_tags + y_pred_tags)
cm = confusion_matrix(y_true_tags, y_pred_tags, labels=labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Test Confusion Matrix - POS Tags (word+char BiLSTM-SDE AR(1) on h_t)")
plt.savefig("/kaggle/working/Test_Confusion_Matrix_word_char_BiLSTM_SDE_AR1.png")
plt.show()

report = classification_report(y_true_tags, y_pred_tags, output_dict=True)
f1s = [report[tag]["f1-score"] for tag in labels if tag != "[PAD]"]
plt.figure(figsize=(12, 6))
plt.bar(labels[1:], f1s)
plt.title("Test F1-score per POS Tag (word+char BiLSTM-SDE AR(1) on h_t)")
plt.xlabel("POS Tag")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("/kaggle/working/Test_F1-score_word_char_BiLSTM_SDE_AR1.png")
plt.show()

sequence_correct = (y_true == y_pred).all(axis=1)
print("Test Sequence-level accuracy:", sequence_correct.mean())

def compute_entropy(prob_vec):
    prob_vec = np.clip(prob_vec, 1e-12, 1 - 1e-12)
    return -np.sum(prob_vec * np.log(prob_vec), axis=-1)

entropies = compute_entropy(y_probs)
plt.figure(figsize=(12, 4))
plt.hist(entropies[mask], bins=50)
plt.title("Token-level Prediction Uncertainty (word+char BiLSTM-SDE AR(1) on h_t)")
plt.xlabel("Entropy")
plt.ylabel("Count")
plt.savefig("/kaggle/working/Test_Token-level_Prediction_Uncertainty_word_char_BiLSTM_SDE_AR1.png")
plt.show()

# --- BLEU Evaluation ---
bleu_hyps, bleu_refs = [], []
for i in range(y_true.shape[0]):
    ref_seq = [idx2tag[tag] for tag in y_true[i] if tag != 0]
    hyp_seq = [idx2tag[tag] for tag in y_pred[i] if tag != 0]
    if ref_seq and hyp_seq:
        bleu_refs.append([ref_seq])
        bleu_hyps.append(hyp_seq)
bleu = corpus_bleu(bleu_refs, bleu_hyps, smoothing_function=SmoothingFunction().method4)
print(f"BLEU score (corpus-level): {bleu*100:.2f}")

# --- Sample Print ---
n_samples = 5
idx2word = {v: k for k, v in word_tokenizer.word_index.items()}
idx2word[0] = "[PAD]"

for i in range(n_samples):
    idx = np.random.randint(len(Xw_test))
    words = [idx2word[w] for w in Xw_test[idx] if w != 0]
    true_seq = [idx2tag[tag] for tag in y_true[idx] if tag != 0]
    pred_seq = [idx2tag[tag] for tag in y_pred[idx] if tag != 0]
    print(f"\nSample {i+1}:")
    print("Words: ", words)
    print("Gold tags:  ", true_seq)
    print("Predicted:  ", pred_seq)
    print("-" * 40)
