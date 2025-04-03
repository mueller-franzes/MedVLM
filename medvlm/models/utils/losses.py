import torch 
import torch.nn as nn
import torch.nn.functional as F 
import math 


class CLIPLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.temperature = nn.Parameter(torch.tensor([1.0]))
        logit_scale = math.log(1 / 0.07)
        self.logit_scale = nn.Parameter(logit_scale * torch.ones([])) 

    # def forward(self, image_embeddings, text_embeddings):
    #     self.logit_scale.data.clamp_(math.log(1), math.log(100))
    #     temperature = torch.exp(self.logit_scale)
    
    #     # https://github.com/moein-shariatnia/OpenAI-CLIP
    #     logits = temperature * text_embeddings @ image_embeddings.T 
    #     images_similarity = image_embeddings @ image_embeddings.T
    #     texts_similarity = text_embeddings @ text_embeddings.T
    #     targets = F.softmax(
    #         (images_similarity + texts_similarity) / 2 * temperature, dim=-1
    #     )
    #     texts_loss = self.cross_entropy(logits, targets, reduction='none')
    #     images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
    #     loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
    #     return loss.mean(), texts_loss.mean(), images_loss.mean()
    
    def forward(self, image_features, text_features, temperature=None):
        # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
        # https://github.com/lucidrains/CoCa-pytorch/blob/edee92c74e311ccfa4a0024412fd991c98aff5fd/coca_pytorch/coca_pytorch.py#L507
        self.logit_scale.data.clamp_(math.log(1), math.log(100))
        temperature = torch.exp(self.logit_scale) if temperature is None else temperature

        logits = temperature * image_features @ text_features.T
    
        device = logits.device
        labels = torch.arange(logits.shape[0], device=device, dtype=torch.long)


        images_loss = F.cross_entropy(logits, labels)
        texts_loss = F.cross_entropy(logits.T, labels)
        total_loss = (images_loss + texts_loss) / 2

        return total_loss, texts_loss, images_loss


    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()