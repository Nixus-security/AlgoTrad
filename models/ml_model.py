"""
LSTM + Attention model for directional price prediction.
Returns (direction, confidence) where confidence ∈ [0,1].
"""
from __future__ import annotations
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from utils.logger import logger

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "cache", "lstm_model.keras")


# ── Attention layer ───────────────────────────────────────────────────────────
class BahdanauAttention(layers.Layer):
    def __init__(self, units: int, **kw):
        super().__init__(**kw)
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features):
        score = tf.nn.tanh(self.W(features))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context = attention_weights * features
        return tf.reduce_sum(context, axis=1)


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(timesteps: int, n_features: int, lstm_units: list[int], dropout: float) -> Model:
    inp = layers.Input(shape=(timesteps, n_features))
    x = layers.LSTM(lstm_units[0], return_sequences=True, dropout=dropout)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(lstm_units[1], return_sequences=True, dropout=dropout)(x)
    x = BahdanauAttention(64)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)  # P(up)
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class MLPredictor:
    def __init__(self, cfg: dict):
        self.cfg = cfg["ml"]
        self.model: Model | None = None
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(
                    MODEL_PATH, custom_objects={"BahdanauAttention": BahdanauAttention}
                )
                logger.info("Loaded cached LSTM model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    # ── Train with time-series cross-validation (anti-overfitting) ───────────
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        timesteps, n_features = X.shape[1], X.shape[2]
        cfg = self.cfg
        n_splits = 3
        tscv = TimeSeriesSplit(n_splits=n_splits)
        val_accuracies = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold+1}/{n_splits}")
            model = build_model(timesteps, n_features,
                                cfg["lstm_units"], cfg["dropout"])

            # Class weights — balance UP/DOWN even if threshold filtering skewed distribution
            n_pos = int(y[tr_idx].sum())
            n_neg = len(tr_idx) - n_pos
            total = n_pos + n_neg
            cw = {0: total / (2 * n_neg + 1e-9), 1: total / (2 * n_pos + 1e-9)}

            model.fit(
                X[tr_idx], y[tr_idx],
                validation_data=(X[val_idx], y[val_idx]),
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                class_weight=cw,
                callbacks=[
                    EarlyStopping(patience=8, restore_best_weights=True,
                                  monitor="val_accuracy"),
                    ReduceLROnPlateau(patience=4, factor=0.4, min_lr=1e-5,
                                      monitor="val_accuracy"),
                ],
                verbose=0,
            )
            _, acc = model.evaluate(X[val_idx], y[val_idx], verbose=0)
            val_accuracies.append(acc)
            logger.info(f"Fold {fold+1} val_accuracy={acc:.4f}")

        # Retrain final model on all data
        self.model = build_model(timesteps, n_features,
                                 cfg["lstm_units"], cfg["dropout"])
        self.model.fit(X, y, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                       callbacks=[EarlyStopping(monitor="loss", patience=7, restore_best_weights=True)],
                       verbose=0)
        self.model.save(MODEL_PATH)
        logger.info(f"Model saved — mean CV acc={np.mean(val_accuracies):.4f}")
        return {"cv_accuracy": np.mean(val_accuracies),
                "cv_std": np.std(val_accuracies)}

    # ── Predict on single sequence ────────────────────────────────────────────
    def predict(self, sequence: np.ndarray) -> tuple[str, float]:
        """
        sequence: shape (1, timesteps, n_features)
        Returns (direction, confidence)
        """
        if self.model is None:
            logger.warning("Model not trained — returning NEUTRAL")
            return "NEUTRAL", 0.5
        prob_up = float(self.model.predict(sequence, verbose=0)[0][0])
        if prob_up >= 0.6:
            return "BUY", prob_up
        if prob_up <= 0.4:
            return "SELL", 1.0 - prob_up
        return "NEUTRAL", max(prob_up, 1.0 - prob_up)
