import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model._abstract_model import SequentialRecModel


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()
        hidden_size = args.hidden_size
        inner_size = 4 * args.hidden_size

        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(args.hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation,
                     groups=groups, bias=bias)

def get_bn(channels):
    return nn.BatchNorm1d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result

class LargeKernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, small_kernel=None):
        super(LargeKernel, self).__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        self.lkb_origin = conv_bn(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=1,
                                groups=groups,
                                bias=False)
    
    def forward(self, x):
        return self.lkb_origin(x)

class ConvInterBlock(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x

class ModernTCNBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, drop=0.1):
        super(ModernTCNBlock, self).__init__()
        
        self.dw_conv = LargeKernel(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            groups=hidden_size
        )
        
        self.ffn1 = ConvInterBlock(
            in_features=hidden_size, 
            hidden_features=hidden_size * 4, 
            drop=drop
        )
        
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(drop)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x

        x = x.transpose(1, 2)
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm1(x + residual)

        residual = x
        x = self.ffn1(x)
        x = self.norm2(x + residual)

        residual = x
        x = self.ffn2(x)
        x = self.norm3(x + residual)

        return x

class EMA(nn.Module):
    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to(x.device)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)

class TWBlock(nn.Module):
    def __init__(self, args):
        super(TWBlock, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_hidden_layers
        
        kernel_size = min(15, args.max_seq_length // 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.temporal_memory_emas = nn.ModuleList([
            EMA(alpha=0.1) for _ in range(self.num_layers)
        ])
        self.temporal_memory_linears = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)
        ])
        
        self.context_convs = nn.ModuleList([
            ModernTCNBlock(self.hidden_size, kernel_size, args.hidden_dropout_prob) 
            for _ in range(self.num_layers)
        ])
        
        self.fusion_layers = nn.ModuleList([
            nn.Linear(self.hidden_size * 2, self.hidden_size) for _ in range(self.num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, hidden_states, attention_mask=None, output_all_encoded_layers=False):
        all_hidden_states = [hidden_states]
        
        for i in range(self.num_layers):
            temporal_memory = self.temporal_memory_emas[i](hidden_states)
            temporal_memory = self.temporal_memory_linears[i](temporal_memory)
            
            context = self.context_convs[i](hidden_states)
            
            combined = torch.cat([temporal_memory, context], dim=-1)
            fused = self.fusion_layers[i](combined)
            
            hidden_states = self.layer_norms[i](fused + hidden_states)
            all_hidden_states.append(hidden_states)
        
        if output_all_encoded_layers:
            return all_hidden_states
        
        return self.final_norm(hidden_states)

class TimeWeaverModel(SequentialRecModel):
    def __init__(self, args):
        super(TimeWeaverModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = TimeWeaverEncoder(args)

        self.time_embed = nn.Sequential(
            nn.Embedding(args.max_time_seq, args.hidden_size),
            nn.Dropout(args.hidden_dropout_prob)
        )

        self.time_scale = nn.Parameter(torch.tensor(0.1))
        
        self.apply(self.init_weights)

        self.time_dropout = nn.Dropout(args.hidden_dropout_prob)

    def add_position_embedding(self, input_ids, time_ids=None):
        seq_length = input_ids.size(1)
        
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_emb = self.position_embeddings(position_ids)
        
        if time_ids is not None:
            time_emb = self.time_embed(time_ids % self.args.max_time_seq)
            time_emb = self.time_dropout(time_emb)
            position_emb = position_emb + self.time_scale * time_emb.mean(dim=0)
        
        item_emb = self.item_embeddings(input_ids)
        
        sequence_emb = item_emb + position_emb.unsqueeze(0)
        sequence_emb = self.LayerNorm(sequence_emb)
        return self.dropout(sequence_emb)

    def forward(self, input_ids, time_ids=None, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        
        sequence_emb = self.add_position_embedding(input_ids, time_ids)
        
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, time_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids, time_ids, user_ids)
        seq_output = seq_output[:, -1, :]

        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)
        return loss
    
    def predict(self, input_ids, time_ids, user_ids=None):
        return self.forward(input_ids, time_ids, user_ids, all_sequence_output=False)
    
class TimeWeaverEncoder(nn.Module):
    def __init__(self, args):
        super(TimeWeaverEncoder, self).__init__()
        self.args = args
        block = TimeWeaverBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class TimeWeaverBlock(nn.Module):
    def __init__(self, args):
        super(TimeWeaverBlock, self).__init__()
        self.layer = TimeWeaverLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class TimeWeaverLayer(nn.Module):
    def __init__(self, args):
        super(TimeWeaverLayer, self).__init__()
        self.args = args
        self.TWBlock = TWBlock(args)

    def forward(self, input_tensor, attention_mask):
        output = self.TWBlock(input_tensor, attention_mask, output_all_encoded_layers=False)
        return output