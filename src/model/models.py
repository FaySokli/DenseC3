import torch
from torch import clamp as t_clamp
from torch import nn as nn
from torch import sum as t_sum
from torch import max as t_max
from torch import einsum
from torch.nn import functional as F

class Specializer(nn.Module):
    def __init__(self, hidden_size, device):
        super(Specializer, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        self.embedding_changer_1 = nn.Linear(self.hidden_size, self.hidden_size//2).to(device)
        self.embedding_changer_4 = nn.Linear(self.hidden_size//2, self.hidden_size).to(device)
        
    def forward(self, embs):
        embs_1 = F.relu(self.embedding_changer_1(embs))
        embs_2 = self.embedding_changer_4(embs_1)
        
        return embs_2


class MoEBiEncoder(nn.Module):
    def __init__(
        self,
        doc_model,
        tokenizer,
        num_classes,
        max_tokens,
        normalize,
        specialized_mode,
        pooling_mode,
        use_adapters,
        device,
    ):
        super(MoEBiEncoder, self).__init__()
        self.doc_model = doc_model.to(device)
        self.hidden_size = self.doc_model.config.hidden_size
        self.tokenizer = tokenizer
        self.device = device
        self.normalize = normalize
        self.max_tokens = max_tokens
        self.use_adapters = use_adapters
        assert specialized_mode in ['sbmoe_top1', 'sbmoe_all'], 'Only sbmoe_top1 and sbmoe_all specialzed mode allowed'
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


    def encoder_no_moe(self, sentences):
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
        
    
    def encoder(self, sentences, logits):
    # def encoder(self, sentences):
        embedding = self.encoder_no_moe(sentences)
        if self.use_adapters:
            # logits = self.cls(embedding).to(self.device)
            embedding = self.embedder(embedding, logits)
        return embedding
    
    def cls(self, out):
    # def cls(self, embedding):
    #     x1 = F.relu(self.cls_1(embedding))
    #     # x2 = F.relu(self.cls_2(x1))
    #     out = self.cls_3(x1)

        # if self.training:
        #     noise_logits = self.noise_linear(embedding)
        #     noise = torch.randn_like(out)*F.softplus(noise_logits)
        #     noisy_logits = out + noise

        #     noisy_logits = torch.softmax(noisy_logits, dim=-1)

        #     # TOP-k GATING
        #     topk_values, topk_indices = torch.topk(noisy_logits, 1, dim=1)
        #     mask = torch.zeros_like(noisy_logits).scatter_(1, topk_indices, 1)
            
        #     # Multiply the original output with the mask to keep only the max value
        #     noisy_logits = noisy_logits * mask
        #     return noisy_logits
        
        # else:
        if self.specialized_mode == 'sbmoe_top1':
            out = torch.softmax(out, dim=-1)

            # TOP-k GATING
            topk_values, topk_indices = torch.topk(out, 1, dim=1)
            mask = torch.zeros_like(out).scatter_(1, topk_indices, 1)
            
            # Multiply the original output with the mask to keep only the max value
            out = out * mask
            return out
        
        elif self.specialized_mode == 'sbmoe_all':
            out = torch.softmax(out, dim=-1)
            return out
    

    def forward(self, data):
        logits_class = self.cls(data[2]).to(self.device)
        query_embedding = self.encoder(data[0], logits_class)
        pos_embedding = self.encoder(data[1], logits_class)
        # query_embedding = self.encoder(data[0], data[2])
        # pos_embedding = self.encoder(data[1], data[2])
        # query_embedding = self.encoder(data[0])
        # pos_embedding = self.encoder(data[1])

        return query_embedding, pos_embedding

    def val_forward(self, data):
        logits_class = self.cls(data[2]).to(self.device)
        query_embedding = self.encoder(data[0], logits_class)
        pos_embedding = self.encoder(data[1], logits_class)
        # query_embedding = self.encoder(data[0], data[2])
        # pos_embedding = self.encoder(data[1], data[2])
        # query_embedding = self.encoder(data[0])
        # pos_embedding = self.encoder(data[1])

        return query_embedding, pos_embedding


    def embedder(self, embedding, logits_class):
        embs = [self.specializer[i](embedding) for i in range(self.num_classes)]

        embs = torch.stack(embs, dim=1)
        
        embs = F.normalize(einsum('bmd,bm->bd', embs, logits_class), dim=-1, eps=1e-6) + embedding

        if self.normalize:
            return F.normalize(embs, dim=-1)
        return embs
    
    def embedder_q(self, embedding):
        embs = [self.specializer[i](embedding) for i in range(self.num_classes)]
        embs = torch.stack(embs, dim=1)
        
        # return F.normalize(embs, dim=-1)
        aggregated_embs = torch.mean(embs, dim=1)
        aggregated_embs = F.normalize(aggregated_embs, dim=-1) + embedding

        if self.normalize:
            aggregated_embs = F.normalize(aggregated_embs, dim=-1)
        return aggregated_embs
    
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
