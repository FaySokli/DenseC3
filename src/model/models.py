import torch
from torch import clamp as t_clamp
from torch import nn as nn
from torch import sum as t_sum
from torch import max as t_max
from torch import sigmoid, softmax
from torch import einsum
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoConfig, BertModel, BertLayer, AutoModel
# from functorch import combine_state_for_ensemble
import copy

class BiEncoder(nn.Module):
    def __init__(
        self,
        doc_model,
        tokenizer,
        max_tokens=512,
        normalize=False,
        pooling_mode='mean',
        device='cpu', 
    ):
        super(BiEncoder, self).__init__()
        self.doc_model = doc_model.to(device)
        self.hidden_size = self.doc_model.config.hidden_size
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.device = device
        self.normalize = normalize
        
        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        if pooling_mode == 'mean':
            self.pooling = self.mean_pooling
        elif pooling_mode == 'max':
            self.pooling = self.max_pooling
        elif pooling_mode == 'cls':
            self.pooling = self.cls_pooling
        elif pooling_mode == 'identity':
            self.pooling = self.identity
        
        
        
    def query_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])
        
    
    def doc_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])
        
    
    def forward(self, data):
        query_embedding = self.query_encoder(data[0])
        pos_embedding = self.doc_encoder(data[1])
        
        return query_embedding, pos_embedding

        
    def val_forward(self, data):
        with torch.no_grad():
            query_embedding = self.query_encoder(data[0])
            pos_embedding = self.doc_encoder(data[1])
        
        return query_embedding, pos_embedding


    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)


    @staticmethod
    def cls_pooling(model_output, attention_mask):
        last_hidden = model_output["last_hidden_state"]
        # last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden[:, 0]


    @staticmethod
    def identity(model_output, attention_mask):
        return model_output['pooler_output']
    

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]


class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, num_experts_to_use=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.num_experts_to_use = num_experts_to_use
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                # nn.LayerNorm(output_dim, eps=1e-12)
            ) for _ in range(num_experts)
        ])
        
        self.meta_expert = copy.deepcopy(self.experts[0])
        # self.meta_expert.to('meta')
        #self.experts = self._init_experts(num_experts, input_dim, hidden_dim, output_dim)

        

    def forward(self, x, gating_weights):
        
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-1])
        
        
        # output = torch.zeros(x.shape[0], self.experts[0][-1].out_features, device=x.device)
        # fmodel, params, buffers = combine_state_for_ensemble(self.experts)
        # [p.requires_grad_() for p in params]
        
        
        # [p.requires_grad_() for p in params];
        # expert_outputs = torch.vmap(fmodel, in_dims=(0, 0, None))(params, buffers, x)
        
        if self.eval():
            def wrapper(params, buffers, data):
                return torch.func.functional_call(self.meta_expert, (params, buffers), data)
            params, buffers = torch.func.stack_module_state(self.experts)
            expert_outputs = torch.vmap(wrapper, in_dims=(0, 0, None))(params, buffers, x)
        else:
            expert_outputs = torch.stack([expert(x) for expert in self.experts])
        
        gating_weights = torch.ones_like(gating_weights)
        if self.num_experts_to_use == self.num_experts:
            output = torch.einsum('ij,jik->ik', gating_weights, expert_outputs)
            
            if len(original_shape) > 2:
                output = output.view(*original_shape[:-1], -1)

            return output

        """
        top_k_weights, top_k_indices = torch.topk(gating_weights, self.num_experts_to_use, dim=-1)
        output = torch.zeros_like(x)
        for i in range(self.num_experts_to_use):
            selected_outputs = expert_outputs[top_k_indices[:, i], torch.arange(x.shape[0])]
            output += selected_outputs * top_k_weights[:, i].unsqueeze(-1)

        output = x + output
        
        # Reshape output back to original shape if necessary
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)
        return output
        """

class MoEBertLayer(BertLayer):
    def __init__(self, config, num_experts=4, num_experts_to_use=4):
        super(MoEBertLayer, self).__init__(config)
        self.moe = MoE(config.hidden_size, config.hidden_size//4, config.hidden_size, num_experts, num_experts_to_use)
        # self.out_normalize = nn.LayerNorm(config.hidden_size)
        self.gating_network = nn.Linear(config.hidden_size, num_experts)
        self.normalizer = nn.LayerNorm(config.hidden_size, eps=1e-12)


    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        first_output = super(MoEBertLayer, self).forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        outputs = first_output[1:]

        
        gating_weights = self.gating_network(first_output[0].mean(1))
        gating_weights = torch.softmax(gating_weights, dim=-1)
        pad_size = first_output[0].shape[1]
        gating_weights = gating_weights.repeat_interleave(pad_size, dim=0)
        
        
        layer_output = self.moe(first_output[0], gating_weights)
        # layer_output = first_output[0]
        layer_output = layer_output + first_output[0]
        layer_output = self.normalizer(layer_output)
        
        outputs = (layer_output,) + outputs
        return outputs

class BertWithMoE(nn.Module):
    def __init__(self, model_name, num_experts=4, num_experts_to_use=2, train_only_moe=True):
        super(BertWithMoE, self).__init__()
        self.num_experts = num_experts
        self.num_experts_to_use = num_experts_to_use
        self.train_only_moe = train_only_moe
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.real_bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        self._init_weights()

    def _init_weights(self):
        self.bert.encoder.layer = nn.ModuleList([
            MoEBertLayer(
                self.config, 
                self.num_experts, 
                self.num_experts_to_use, 
            )
            for _ in range(self.config.num_hidden_layers)
        ])
        for i, _ in enumerate(self.bert.encoder.layer):
            self.bert.encoder.layer[i].attention = self.real_bert.encoder.layer[i].attention
            self.bert.encoder.layer[i].intermediate = self.real_bert.encoder.layer[i].intermediate
            self.bert.encoder.layer[i].output = self.real_bert.encoder.layer[i].output
            
            for expert in self.bert.encoder.layer[i].moe.experts:
                expert[0].weight.data.normal_(mean=0.0, std=0.02)
                expert[0].bias.data.zero_()
                
                expert[2].weight.data.normal_(mean=0.0, std=0.02)
                expert[2].bias.data.zero_()
                
                # expert[3].bias.data.zero_()
                # expert[3].weight.data.fill_(1.0)

            if self.train_only_moe:
                self.bert.encoder.layer[i].attention.self.query.weight.requires_grad = False
                self.bert.encoder.layer[i].attention.self.query.bias.requires_grad = False
                self.bert.encoder.layer[i].attention.self.key.weight.requires_grad = False
                self.bert.encoder.layer[i].attention.self.key.bias.requires_grad = False
                self.bert.encoder.layer[i].attention.self.value.weight.requires_grad = False
                self.bert.encoder.layer[i].attention.self.value.bias.requires_grad = False
                self.bert.encoder.layer[i].attention.output.dense.weight.requires_grad = False
                self.bert.encoder.layer[i].attention.output.dense.bias.requires_grad = False
                self.bert.encoder.layer[i].attention.output.LayerNorm.weight.requires_grad = False
                self.bert.encoder.layer[i].attention.output.LayerNorm.bias.requires_grad = False
                
                self.bert.encoder.layer[i].intermediate.dense.weight.requires_grad = False
                self.bert.encoder.layer[i].intermediate.dense.bias.requires_grad = False
                
                self.bert.encoder.layer[i].output.dense.weight.requires_grad = False
                self.bert.encoder.layer[i].output.dense.bias.requires_grad = False
                self.bert.encoder.layer[i].output.LayerNorm.weight.requires_grad = False
                self.bert.encoder.layer[i].output.LayerNorm.bias.requires_grad = False
        
        self.bert.embeddings = self.real_bert.embeddings
        self.bert.pooler = self.real_bert.pooler
        if self.train_only_moe:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
            self.bert.embeddings.LayerNorm.weight.requires_grad = False
            self.bert.embeddings.LayerNorm.bias.requires_grad = False
            
            self.bert.pooler.dense.weight.requires_grad = False
            self.bert.pooler.dense.bias.requires_grad = False

        self.real_bert = None


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


class Specializer(nn.Module):
    def __init__(self, hidden_size, device):
        super(Specializer, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        self.query_embedding_changer_1 = nn.Linear(self.hidden_size, self.hidden_size//2).to(device)
        self.query_embedding_changer_4 = nn.Linear(self.hidden_size//2, self.hidden_size).to(device)
        
    def forward(self, query_embs):
        query_embs_1 = F.relu(self.query_embedding_changer_1(query_embs))
        query_embs_2 = self.query_embedding_changer_4(query_embs_1)
        
        return query_embs_2


class MoEBiEncoder(nn.Module):
    def __init__(
        self,
        doc_model,
        tokenizer,
        num_classes,
        max_tokens=512,
        normalize=False,
        specialized_mode='sbmoe',
        pooling_mode='mean',
        use_adapters=True,
        device='cpu',
    ):
        super(MoEBiEncoder, self).__init__()
        self.doc_model = doc_model.to(device)
        self.hidden_size = self.doc_model.config.hidden_size
        self.tokenizer = tokenizer
        self.device = device
        self.normalize = normalize
        self.max_tokens = max_tokens
        self.use_adapters = use_adapters
        assert specialized_mode in ['sbmoe', 'zeros', 'ones', 'rand'], 'Only sbmoe, zeros, ones and rand specialzed mode allowed'
        self.specialized_mode = specialized_mode
        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        if pooling_mode == 'mean':
            self.pooling = self.mean_pooling
        elif pooling_mode == 'max':
            self.pooling = self.max_pooling
        elif pooling_mode == 'cls':
            self.pooling = self.cls_pooling
        elif pooling_mode == 'identity':
            self.pooling = self.identity
        
        self.num_classes = num_classes
        self.init_cls()
        
        self.specializer = nn.ModuleList([Specializer(self.hidden_size, self.device) for _ in range(self.num_classes)])    
        
        
    def query_encoder_no_moe(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])
        
    
    def init_cls(self):
        self.cls_1 = nn.Linear(self.hidden_size, self.hidden_size//2).to(self.device)
        # self.cls_2 = nn.Linear(self.hidden_size*2, self.hidden_size*4).to(self.device)
        self.cls_3 = nn.Linear(self.hidden_size//2, self.num_classes).to(self.device)
        self.noise_linear = nn.Linear(self.hidden_size, self.num_classes).to(self.device)
        
    
    def query_encoder(self, sentences):
        query_embedding = self.query_encoder_no_moe(sentences)
        if self.use_adapters:
            query_class = self.cls(query_embedding)
            query_embedding = self.query_embedder(query_embedding, query_class)
        return query_embedding
        

    def doc_encoder_no_moe(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])

    def doc_encoder(self, sentences):
        doc_embedding = self.doc_encoder_no_moe(sentences)
        if self.use_adapters:
            doc_class = self.cls(doc_embedding)
            doc_embedding = self.doc_embedder(doc_embedding, doc_class)
        return doc_embedding
        

    def cls(self, query_embedding):
        x1 = F.relu(self.cls_1(query_embedding))
        # x2 = F.relu(self.cls_2(x1))
        out = self.cls_3(x1)

        if self.training:
            #Adding scaled unit gaussian noise to the logits
            noise_logits = self.noise_linear(query_embedding)
            noise = torch.randn_like(out)*F.softplus(noise_logits)
            noisy_logits = out + noise

            noisy_logits = torch.softmax(noisy_logits, dim=-1)

            # TOP-k GATING
            topk_values, topk_indices = torch.topk(noisy_logits, 1, dim=1)
            mask = torch.zeros_like(noisy_logits).scatter_(1, topk_indices, 1)
            
            # Multiply the original output with the mask to keep only the max value
            noisy_logits = noisy_logits * mask
            return noisy_logits
        
        else:
            out = torch.softmax(out, dim=-1)

            # TOP-k GATING
            topk_values, topk_indices = torch.topk(out, 1, dim=1)
            mask = torch.zeros_like(out).scatter_(1, topk_indices, 1)
            
            # Multiply the original output with the mask to keep only the max value
            out = out * mask
            return out
    

    def forward(self, data):
        query_embedding = self.query_encoder(data[0])
        pos_embedding = self.doc_encoder(data[1])

        return query_embedding, pos_embedding

    def val_forward(self, data):
        query_embedding = self.query_encoder(data[0])
        pos_embedding = self.doc_encoder(data[1])

        return query_embedding, pos_embedding


    def query_embedder(self, query_embedding, query_class):
        query_embs = [self.specializer[i](query_embedding) for i in range(self.num_classes)]
        
        query_embs = torch.stack(query_embs, dim=1)
        
        if self.specialized_mode == 'sbmoe':
            query_class = query_class #sigmoid(query_class) # softmax(query_class, dim=-1)
        if self.specialized_mode == 'zeros':
            query_class = torch.zeros(query_class.shape).to(self.device)
        if self.specialized_mode == 'ones':
            query_class = torch.ones(query_class.shape).to(self.device)
        if self.specialized_mode == 'rand':
            query_class = torch.rand(query_class.shape).to(self.device)

        query_embs = F.normalize(einsum('bmd,bm->bd', query_embs, query_class), dim=-1, eps=1e-6) + query_embedding

        if self.normalize:
            return F.normalize(query_embs, dim=-1)
        return query_embs
    
    def doc_embedder(self, doc_embedding, doc_class):
        return self.query_embedder(doc_embedding, doc_class)
        

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)


    @staticmethod
    def cls_pooling(model_output, attention_mask):
        last_hidden = model_output["last_hidden_state"]
        # last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden[:, 0]


    @staticmethod
    def identity(model_output, attention_mask):
        return model_output['pooler_output']
    

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]
