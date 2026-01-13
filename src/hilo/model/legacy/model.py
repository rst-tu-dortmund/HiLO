import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from hilo.model.base_model import BaseModel
from hilo.model.legacy.utils.sinusoidal_embedding import SinusoidalEmbedding
from hilo.model.legacy.modules.decoder import Decoder, FusionDecoder
from hilo.model.legacy.modules.encoder import Encoder, SensorEncoder
from hilo.model.legacy.modules.predict import MLPModule, MLP

from einops import rearrange, repeat
from torch import Tensor
from typing import Optional


class Transformer(BaseModel):
    def __init__(
        self,
        cfg
    ):  # input with std_dev has 18 params, and without std_dev 11 params
        super().__init__(cfg)
        self.cfg = cfg
        self.max_seq_len = cfg["max_seq_len"]
        self.d_model = cfg["d_model"]
        self.posEmbed = SinusoidalEmbedding(
            2, 2
        )  # 2 inputs channels x and y, 2 frequencies.
        self.MLPInputChannelSize = (
            cfg["param_size"] + 8
        )  # 2 inputs channels x and y, 2 frequencies.
        # self.inputEmbed = MLPModule(input_size = 26, hidden_sizes= [16, 32, 64],    # inputsize = param_size + ( Inputchannels_sinEmbed(2*Num_Frequency))
        #                             output_size=self.d_model)
        self.inputEmbed = MLPModule(
            input_size=self.MLPInputChannelSize,
            hidden_sizes=[
                32,
                32,
                64,
            ],  # inputsize = param_size + ( Inputchannels_sinEmbed(2*Num_Frequency))
            output_size=self.d_model,
            batch_size=cfg["batch_size"],
            max_seq_len=cfg["max_seq_len"],
            num_sensors=cfg["num_sensors"],
        )  # without bottleneck layer
        self.num_of_sensors = cfg["num_sensors"]
        self.number_of_queries = cfg["max_seq_len"]
        self.query_embed = nn.Embedding(cfg["max_seq_len"], self.d_model)
        self.nhead = cfg["nhead"]
        self.bboxHead = MLP(self.d_model, self.d_model, 8, 3, dropout=cfg["bbox_drop"])
        self.class_head_dropout = nn.Dropout(cfg["cls_drop"], inplace=False)
        self.classHead = nn.Linear(self.d_model, cfg["num_classes"] + 1)
        self.sensorEncoder = SensorEncoder(
            self.d_model, cfg["dim_feedforward"], cfg["num_encoder_layers"], cfg["nhead"], use_original=cfg["use_original_encoder"]
        )
        self.fusiondecoder = FusionDecoder(
            self.d_model, cfg["dim_feedforward"], cfg["num_decoder_layers"], cfg["nhead"]
        )
        
        # Normalizing ranges
        xRange = cfg["norm"]["x_range"]
        self.register_buffer("xRange", torch.tensor(xRange))
        yRange = cfg["norm"]["y_range"]
        self.register_buffer("yRange", torch.tensor(yRange))
        lRange = cfg["norm"]["l_range"]
        self.register_buffer("lRange", torch.tensor(lRange))
        wRange = cfg["norm"]["w_range"]
        self.register_buffer("wRange", torch.tensor(wRange))
        vRange = cfg["norm"]["v_range"]
        self.register_buffer("vRange", torch.tensor(vRange))
        hRange = cfg["norm"]["h_range"]
        self.register_buffer("hRange", torch.tensor(hRange))
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def normalize_data(self, data):
        
        # data: [B, S, O, 18] (incl. std_dev)
        x_norm = (data[..., 0] - self.xRange[0]) / (self.xRange[1] - self.xRange[0])
        y_norm = (data[..., 1] - self.yRange[0]) / (self.yRange[1] - self.yRange[0])
        l_norm = (data[..., 2] - self.lRange[0]) / (self.lRange[1] - self.lRange[0])
        w_norm = (data[..., 3] - self.wRange[0]) / (self.wRange[1] - self.wRange[0])
        V_norm_0 = (data[..., 4] - self.vRange[0]) / (self.vRange[1] - self.vRange[0])
        V_norm_1 = (data[..., 5] - self.vRange[0]) / (self.vRange[1] - self.vRange[0])
        h_norm = (data[..., 6] - self.hRange[0]) / (self.hRange[1] - self.hRange[0])
        
        x_std_norm = (data[..., 9] - self.xRange[0]) / (self.xRange[1] - self.xRange[0])
        y_std_norm = (data[..., 10] - self.yRange[0]) / (self.yRange[1] - self.yRange[0])
        l_std_norm = (data[..., 11] - self.lRange[0]) / (self.lRange[1] - self.lRange[0])
        w_std_norm = (data[..., 12] - self.wRange[0]) / (self.wRange[1] - self.wRange[0])
        V_std_norm_0 = (data[..., 13] - self.vRange[0]) / (self.vRange[1] - self.vRange[0])
        V_std_norm_1 = (data[..., 14] - self.vRange[0]) / (self.vRange[1] - self.vRange[0])
        h_std_norm = (data[..., 15] - self.hRange[0]) / (self.hRange[1] - self.hRange[0])
        
        data_norm = torch.stack([
            x_norm,
            y_norm,
            l_norm,
            w_norm,
            V_norm_0,
            V_norm_1,
            h_norm,
            data[..., 7],  # class
            data[..., 8],  # score
            x_std_norm,
            y_std_norm,
            l_std_norm,
            w_std_norm,
            V_std_norm_0,
            V_std_norm_1,
            h_std_norm,
            data[..., 16]/4,  # sensor id
            data[..., 17],  # time_to_gt
        ], dim=-1)
        
        return data_norm
    
    def denormalize_data(self, output):
        centers_denorm = torch.stack([
            output["centers"][..., 0] * (self.xRange[1] - self.xRange[0]) + self.xRange[0],
            output["centers"][..., 1] * (self.yRange[1] - self.yRange[0]) + self.yRange[0],
        ], dim=-1)
        extents_denorm = torch.stack([
            output["extents"][..., 0] * (self.lRange[1] - self.lRange[0]) + self.lRange[0],
            output["extents"][..., 1] * (self.wRange[1] - self.wRange[0]) + self.wRange[0],
        ], dim=-1)
        velocities_denorm = torch.stack([
            output["velocities"][..., 0] * (self.vRange[1] - self.vRange[0]) + self.vRange[0],
            output["velocities"][..., 1] * (self.vRange[1] - self.vRange[0]) + self.vRange[0],
        ], dim=-1)
        
        output["centers"] = centers_denorm
        output["extents"] = extents_denorm
        output["velocities"] = velocities_denorm
        
        output["bboxes"] = torch.cat([
            centers_denorm,
            extents_denorm,
            velocities_denorm,
            output["yaws"].unsqueeze(-1)
        ], dim=-1)
        
        return output
        
    def forward(self, x_, mask):

        # stack all the sensor data into a tensor
        # x = sensor_data
        bs, no_of_sensors, no_of_obj, objParam = x_.shape
        x = self.normalize_data(x_)
        x = rearrange(x, "B S O C -> (B S O) C").to(torch.float32)
        # separate positon parameters (x,y) fom rest
        Position = x[:, :2]
        rest = x[:, 2:]
        # sinusoidal embedding the object positions
        posSinEmbed = self.posEmbed(Position)
        x = torch.cat([posSinEmbed, rest], dim=1)
        # # Input embedding the sensor data
        x = self.inputEmbed(
            x
        )  # .view(bs, no_of_sensors, no_of_obj,-1).permute(1,2,0,3)
        # sensor data  to tranformer encoder
        x = rearrange(
            x, "(B S O) C -> (S O) B C", B=bs, S=no_of_sensors, O=no_of_obj
        )  # convert to [100, bs, Cin] so self-attention works on all the sensors
        encoded = self.sensorEncoder(x)

        # Decoding
        query_embed = self.query_embed.weight
        query_embed = repeat(query_embed, "O C -> O B C", B=bs)
        # object_queries = torch.zeros_like(query_embed)  # learnable object queries
        # tgt_attn_mask = torch.triu(torch.ones(object_queries.size(0), object_queries.size(0)), diagonal=1).bool().to('cuda')  # Create attention mask for target
        decoded_output = self.fusiondecoder(query_embed, encoded)

        # Detection
        seq_len, bs, _ = decoded_output.shape
        decoded_output = rearrange(decoded_output, "O B C -> (O B) C")
        # BBox detection
        bbox = self.bboxHead(decoded_output)
        bbox = rearrange(bbox, "(O B) C -> B O C", B=bs, O=self.max_seq_len).sigmoid()
        # class detection
        objClass_ = self.classHead(self.class_head_dropout(decoded_output))
        objClass = rearrange(objClass_, "(O B) C -> B O C", B=bs, O=self.max_seq_len)
        # motion param detection
        # motion = self.motionHead(decoded_output)
        # motion = rearrange(motion, '(O B) C -> B O C', B=bs, O=no_of_obj)

        output_ = self.prepare_output(bbox, objClass)
        output = self.denormalize_data(output_)
        
        return output


def create_model(cfg):
    model = Transformer(
        dropout=cfg["dropout"],
        bbox_drop=cfg["bbox_drop"],
        cls_drop=cfg["cls_drop"],
        max_seq_len=cfg["max_seq_len"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        batch_size=cfg["batch_size"],
        num_sensors=cfg["num_sensors"],
        use_original_encoder=cfg["use_original_encoder"]
    )
    return model
