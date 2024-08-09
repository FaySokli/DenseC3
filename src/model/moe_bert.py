import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertLayer, BertEncoder, BertModel, BertForSequenceClassification 
from typing import Optional, Tuple

def get_nonlin_func(nonlin):
    if nonlin == "tanh":
        return torch.tanh
    elif nonlin == "relu":
        return torch.relu
    elif nonlin == "gelu":
        return nn.functional.gelu
    elif nonlin == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError("Unsupported nonlinearity!")

class BottleneckAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.adapter_input_size = config.hidden_size
        self.adapter_latent_size = config.adapter_latent_size
        self.non_linearity = get_nonlin_func(config.adapter_non_linearity)
        self.residual = config.adapter_residual

        # down projection
        self.down_proj = nn.Linear(self.adapter_input_size, self.adapter_latent_size)
        # up projection
        self.up_proj = nn.Linear(self.adapter_latent_size, self.adapter_input_size)

        self.init_weights()

    def init_weights(self):
        """ Initialize the weights -> so that initially we the whole Adapter layer is a near-identity function """
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        output = self.up_proj(self.non_linearity(self.down_proj(x)))
        if self.residual:
            output = x + output
        return output

class AdapterBertIntermediateMoE(BertIntermediate):
    def __init__(self, config, layer_index):
        super().__init__(config)
        self.intermediate_adapter_moe = nn.ModuleList(
            [
                BottleneckAdapterLayer(config) for _ in range(config.num_experts)
            ]
        )

    def forward(self, hidden_states, gating_weights):
        
        hidden_states = torch.stack([output_adapter(hidden_states) for output_adapter in self.intermediate_adapter_moe], dim=1)
        hidden_states = torch.einsum("ij,ijkl->ikl", gating_weights, hidden_states)

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class AdapterBertOutputMoE(BertOutput):
    def __init__(self, config, layer_index):
        super().__init__(config)
        self.output_moe = nn.ModuleList(
            [
                BottleneckAdapterLayer(config) for _ in range(config.num_experts)
            ]
        )
        

    def forward(self, hidden_states, input_tensor, gating_weights):
        # BERT layers
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # MoE layers
        hidden_states = torch.stack([output_adapter(hidden_states) for output_adapter in self.output_moe], dim=1)
        hidden_states = torch.einsum("ij,ijkl->ikl", gating_weights, hidden_states)
        
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class MoEBertLayer(BertLayer):
    def __init__(self, config, layer_index):
        super().__init__(config)        
        self.output = AdapterBertOutputMoE(config, layer_index)
        self.intermediate = AdapterBertIntermediateMoE(config, layer_index)
        self.gating = nn.Linear(config.hidden_size, config.num_experts)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        
        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        
        return outputs
        


    def feed_forward_chunk(self, attention_output):
        gating_weights = self.gating(attention_output.mean(dim=1))
        gating_weights = torch.softmax(gating_weights, dim=-1)
        
        intermediate_output = self.intermediate(attention_output, gating_weights)
        
        layer_output = self.output(intermediate_output, attention_output, gating_weights)
        
        return layer_output


class MoEBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([MoEBertLayer(config, i) for i in range(config.num_hidden_layers)])
        

class MoEBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = MoEBertEncoder(config)
                
        self.freeze_original_params(config)

    def freeze_original_params(self, config):
        for param in self.parameters():
            param.requires_grad = False

        for i in range(config.num_hidden_layers):
            for experts in self.encoder.layer[i].intermediate.intermediate_adapter_moe:
                for param in experts.parameters():
                    param.requires_grad = True
            for experts in self.encoder.layer[i].output.output_moe:
                for param in experts.parameters():
                    param.requires_grad = True
                    
    def unfreeze_original_params(self, config):
        for param in self.parameters():
            param.requires_grad = True