
import torch.nn as nn
import torch 
from transformers import AutoModel
from math import prod


class VectorQuantizer(nn.Module):
    def __init__(self, in_ch, emb_num, emb_ch, commitment_weight=1):
        super().__init__()
        self.in_ch = in_ch 
        self.emb_num = emb_num
        self.emb_ch = emb_ch
        self.commitment_weight = commitment_weight

        self.codebook = torch.nn.Embedding(emb_num, emb_ch) # emb_ch

        # model = AutoModel.from_pretrained("GerMedBERT/medbert-512")
        # for model.param in model.parameters():
        #     model.param.requires_grad = False
        # self.codebook = model.embeddings.word_embeddings

        self.linear_in = nn.Linear(in_ch, emb_ch, bias=False) if in_ch != emb_ch else nn.Identity()
        self.linear_out = nn.Linear(emb_ch, in_ch, bias=False) if in_ch != emb_ch else nn.Identity()
            
    def forward(self, z, **kwargs):
        # z should be [B, N, C]

        z = self.linear_in(z)
        min_encoding_indices = self._get_indices(z)
        z_q = self.codebook(min_encoding_indices) # [B, N, C]

        # Compute Embedding Loss 
        loss =  self.commitment_weight * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2) 

        z_q = z + (z_q - z).detach()

        z_q = self.linear_out(z_q)
        return z_q, min_encoding_indices, loss
        
    def _get_indices(self, z):
        v = self.codebook.weight.data
        dist = self._comp_distance(z, v)
        return torch.argmin(dist, dim=-1) # [B, N]
    
    def _comp_distance(self, z, v):
        # distances from z to embeddings e: (z - e)^2 = z^2 + e^2 - 2 e * z
        return  (   torch.sum(z**2, dim=-1, keepdim=True) 
                 +  torch.sum(v**2, dim=-1)
                -2* torch.einsum("bnc,cs->bns", z, v.t())
        ) # [B, N, num_embeddings] 
    
    def get_codes_from_indices(self, indices, **kwargs):
        z_q = self.codebook(indices)
        z_q = self.linear_out(z_q)
        return z_q



class ResidualVectorQuantizer(nn.Module):
    def __init__(self, vq_num=3, share_residual_codebooks=False, rq_index_sum=True, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([
            VectorQuantizer(**kwargs)  for i in range(vq_num)
        ])
        emb_num_layer = [layer.emb_num for layer in self.layers]
        self.codebook_cumprod = torch.cumprod(torch.tensor([1]+emb_num_layer[:-1]), dim=0) 
        self.emb_num =  prod(emb_num_layer) if rq_index_sum else emb_num_layer[0]
        self.rq_index_sum = rq_index_sum

        first_vq, *rest_vq = self.layers
        if hasattr(first_vq, 'codebook'):
            if share_residual_codebooks:
                codebook = first_vq.codebook
                for vq in rest_vq:
                    vq.codebook = codebook

        
            self.codebook = nn.ModuleList([layer.codebook for layer in self.layers])
    
        # self.attention = CrazyAttention(emb_num=256, emb_ch=16, seq_len=16*16)

    def forward(self, z):
        z_q = 0.0 
        r = z-z_q    
        loss = 0.0
        index_sum = 0
        index = [] 
        logging_dict = {}
        
        for i, quantizer in enumerate([*self.layers, ]): # self.layers[-1]
            r_q, index_i, loss_i = quantizer(r, logging_dict=logging_dict, i=i)
            logging_dict[f'emb_loss_{i}'] = loss_i

            loss += loss_i
            index_sum += index_i*self.codebook_cumprod[i]
            index.append(index_i)

            z_q = z_q+r_q # = z_q + (z -z_q) ~ z 
            r = z -z_q.detach() # = (r+z_q)-z_q'.detach() = (r+z_q)-(z_q+r_q).detach() = r-r_q.detach() 
            logging_dict[f'zq_error_{i}'] = torch.mean((z_q - z) ** 2) 


        loss = loss/len(self.layers)

        # Add joint reconstruction loss for the current quantizer
        loss += torch.mean((z_q - z) ** 2)  # Reconstruction error up to this stage

            


        index = index_sum if self.rq_index_sum else  torch.cat(index, -1) 

        self.logging_dict = logging_dict
        
        return z_q, index, loss

    def index2indices(self, index):
        indices = [] 
        for base in self.codebook_cumprod.flip(dims=(0,)):
            indices.append(index//base)
            index = index-indices[-1]*base  
        return indices[::-1]

    def get_codes_from_indices(self, indices, rq_emb_sum=True, **kwargs):
        if self.rq_index_sum :
            indices = self.index2indices(indices) 
        
        codes = []
        for vq_i, indices_i in zip(self.layers, indices.chunk(len(self.layers), dim=-1)): 
            codes.append(vq_i.get_codes_from_indices(indices_i))
        
        if rq_emb_sum:
            return torch.stack(codes, dim=0).sum(0)
        return torch.cat(codes, dim=-2) 
    



class LFQ(nn.Module):
    def __init__(self, in_ch=None, emb_ch=16, code_scale = 1, commitment_weight=0, entropy_weight = 0, diversity_gamma=1, **kwargs) -> None:
        super().__init__()
        assert not ((in_ch is None) and (emb_num is None)) , 'either in_ch or emb_num must be specified'
        #assert (emb_num is None) or log2(emb_num).is_integer(), 'codebook size must be a power of 2'

        self.linear_in = nn.Linear(in_ch, emb_ch, bias=False) if in_ch != emb_ch else nn.Identity()
        self.linear_out = nn.Linear(emb_ch, in_ch, bias=False) if in_ch != emb_ch else nn.Identity()

        emb_num = 2**emb_ch 
        self.emb_num = emb_num
        

        self.code_scale = code_scale
        self.commitment_weight = commitment_weight
        self.entropy_weight = entropy_weight
        self.diversity_gamma = diversity_gamma
        self.register_buffer("basis", 2 ** torch.arange(emb_ch), persistent=False) # NOTE: Least sign. bit first

        all_indices = torch.arange(emb_num)
        pseudo_codebook = self.get_codes_from_indices(all_indices)
        self.register_buffer('pseudo_codebook', pseudo_codebook, persistent = False)

 
    def forward(self, z, **kwargs):
        z = self.linear_in(z)
        z_q = self.quantize(z)
        indices = self.get_indices(z_q)
        loss = 0.0 

        # entropy aux loss
        if self.entropy_weight > 0:
            distance = self.comp_distance(z, self.pseudo_codebook)
            prob = distance.softmax(dim = -1)
            per_sample_entropy =  self.entropy(prob).mean()
            avg_prob = prob.mean(dim=[0,1])
            codebook_entropy = self.entropy(avg_prob).mean()
            loss += self.entropy_weight*(per_sample_entropy - self.diversity_gamma * codebook_entropy)

        # Commitment loss 
        if self.commitment_weight>0: 
            loss += self.commitment_weight*F.mse_loss(z, z_q.detach())
        
        z_q = self.linear_out(z_q)
        return z_q, indices, loss 
    
    def entropy(self, prob):
        return torch.sum(-prob * torch.log(torch.clamp(prob, min=1e-5)), dim=-1)
    
    def comp_distance(self, z, v):
        # distances from z to embeddings e: (z - e)^2 = z^2 + e^2 - 2 e * z
        return  (  torch.sum(z**2, dim=-1, keepdim=True) 
                  +  torch.sum(v**2, dim=-1) # Constant 
                -2* torch.einsum("bnc,cs->bns", z, v.t())).sqrt()

    def quantize(self, z):
        codes = torch.ones_like(z) * self.code_scale
        z_q = torch.where(z > 0, codes, -codes)
        z_q = z + (z_q - z).detach()
        return z_q 

    def get_indices(self, z_q):
        return torch.sum((z_q>0)*self.basis, dim=-1)

    def get_codes_from_indices(self, indices, **kwargs):
        bits = ((indices.unsqueeze(-1) & self.basis) != 0).float()
        z_q = (2*bits -1) * self.code_scale # 0-> -code_scale; 1-> +code_scale
        z_q = self.linear_out(z_q)
        return z_q 