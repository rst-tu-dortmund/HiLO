import torch.nn as nn
from hilo.model.legacy.modules.decoder import DecoderLayer
from hilo.model.legacy.modules.predict import PredictionBlock

# Decoder for Sensor Fusion and Prediction
class SensorFusionDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        feedforward_dim,
        vocab_size,
        dropout_prob=0.1,
    ):
        super(SensorFusionDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, num_heads, feedforward_dim, dropout_prob)
                for _ in range(num_layers)
            ]
        )
        self.prediction_block = PredictionBlock(
            embed_dim, vocab_size
        )  # PredictionBlock defined earlier

    def forward(self, sensor1_encoded, sensor2_encoded, ground_truth_encoded):
        fused_encoded = sensor1_encoded + sensor2_encoded + ground_truth_encoded
        for layer in self.layers:
            fused_encoded = layer(
                fused_encoded, fused_encoded
            )  # Self-attention for sensor fusion

        # Generate predictions using the fused representation
        predictions = self.prediction_block(fused_encoded)
        return predictions
