"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import argparse
import numpy as np
import os.path as osp
import pandas as pd
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import tqdm as tqdmtqdm
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from builtins import Exception
import os
import torch
import torch.distributed as dist
from torchvision import transforms

from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ops import binary_focal_loss_with_logits, compute_spatial_encodings

import sys
from hico_list import hico_verb_object_list,hico_verbs,hico_verbs_sentence,hico_verbs_sentence_2
from vcoco_list import vcoco_verbs_sentence
sys.path.append('detr')
from detr.models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
import pdb
import CLIP_models_adapter_prior2
import torchvision
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from transformer_module import TransformerDecoderLayer, TransformerSALayer
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip 
from ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, normalize_boxes, compute_box_distance, generate_patch_grid, compute_attention_mask
import pickle, random
from tqdm import tqdm
from hico_text_label import hico_unseen_index, MAP_AO_TO_HOI, HICO_INTERACTIONS, obj_to_name, HOI_TO_AO, HOI_IDX_TO_ACT_IDX, HOI_IDX_TO_OBJ_IDX, ACT_IDX_TO_ACT_NAME, RARE_HOI_IDX
import hico_text_label
from vcoco_text_label import vcoco_hoi_text_label, MAP_AO_TO_HOI_COCO, HOI_TO_AO_COCO
import cv2
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math
from random import sample
import json
import copy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn.utils.rnn import pad_sequence



# feat_dim = 512
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class RelationNet(nn.Module):
    def __init__(self, feature_size=1536, embed_size=128):
        super(RelationNet, self).__init__()
        self.embe_size = embed_size
        self.fc1 = nn.Linear(feature_size, self.embe_size//2)
        self.fc2 = nn.Linear(feature_size, self.embe_size//2)
        self.g_mlp = nn.Sequential(
            nn.Linear(self.embe_size, self.embe_size // 2),
            nn.ReLU(),
            nn.Linear(self.embe_size // 2, self.embe_size // 2),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(self.embe_size // 2, 1)
        self.sigm = nn.Sigmoid()
        with torch.no_grad():
            nn.init.zeros_(self.fc3.weight)
            nn.init.zeros_(self.fc3.bias)
        
    def forward(self, feat1, feat2):
        feat1 = self.fc1(feat1)
        feat2 = self.fc1(feat2)
        feat1_ex = feat1.unsqueeze(1).repeat(1, feat2.shape[0], 1)
        feat2_ex = feat2.unsqueeze(0).repeat(feat1.shape[0], 1, 1)
        relation_pairs = torch.cat((feat1_ex, feat2_ex), 2)
        relation = self.g_mlp(relation_pairs)
        relation = self.fc3(relation)
        relation = self.sigm(relation)
        # relation = relation.sum(1)
        return relation.squeeze(2)

_tokenizer = _Tokenizer()
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # print("!!!", x.device, layer.weight.device)
            try:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            except:
                x = x.to(layer.weight.device)
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
                # print("!!!", x.device, layer.weight.device, layer.bias.device)
        return x

class Weight_Pred(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear1 = MLP(input_dim=input_dim, hidden_dim=512, output_dim=128, num_layers=2)
        self.drop1 = nn.Dropout()
        self.linear2 = MLP(input_dim=128, hidden_dim=32, output_dim=3, num_layers=2)
    
    def forward(self, x):
        x = self.drop1(self.linear1(x))
        x = self.linear2(x)
        return F.sigmoid(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        '''
        x: L * bs * C, 
        prior[0]: bs * L' * C', padded prior knowledge
        prior[1]: bs * L' (mask of prior knowledge)
        ''' 
        tempa = self.attention(self.ln_1(x))
        x = x + tempa[0]  
        x = x + self.mlp(self.ln_2(x))    
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, adapter: bool=False, adapter_layers: List=[i for i in range(24)], adapter_num_layers: int=1):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)[0]
        # return self.resblocks((x,prior))



class Adapter(nn.Module):
    def __init__(self,
                input_size,
                 dropout=0.1,
                 adapter_scalar="1.0",
                 adapter_num_layers=1,
                 mem_adpt_self = False, 
                 SA_only = False,
                 prior_dim = None,
                 prior_dim_2 = None,
                 down_size = 64,
                 ):
        super().__init__()
        self.n_embd = input_size
        self.down_size = down_size
        self.scale = float(adapter_scalar)

        self.down_proj_mem = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj_mem = nn.Linear(self.down_size, self.n_embd)
        self.adapter_num_layers = adapter_num_layers
        if prior_dim is None:
            prior_dim = input_size
        if down_size*2 <= prior_dim:
            self.down_proj_prior = MLP(prior_dim, down_size*2, down_size, 3)
        else:
            self.down_proj_prior = nn.Linear(prior_dim, down_size)
        if prior_dim_2 is not None:
            self.down_proj_prior_2 = MLP(prior_dim_2, 128, 64, 3)
        # if txt_prior == False:
        #     self.down_proj_prior = MLP(feat_dim, 128, 64, 3)
        # else:
        #     self.down_proj_prior = MLP(input_size, 128, 64, 3)

        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj_mem.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj_mem.weight)
            nn.init.zeros_(self.down_proj_mem.bias)
            nn.init.zeros_(self.up_proj_mem.bias)
    
        if SA_only is False:
            instance_decoder_layer = TransformerDecoderLayer(self.down_size, 2, self.down_size*2,
                                                self.dropout, 'relu', False)
        else:
            instance_decoder_layer = TransformerSALayer(self.down_size, 2, self.down_size*2,
                                                self.dropout, 'relu', False)            
        self.mhsa_layers = _get_clones(instance_decoder_layer, adapter_num_layers)
        self.mem_adpt_self = mem_adpt_self

    def forward(self, x, prior = None, prior_2 = None):
        if x.type() == 'torch.cuda.HalfTensor':
            self.down_proj_mem.half()
        tempa = self.down_proj_mem(x)

        # pdb.set_trace()
        if prior is None or self.mem_adpt_self is True:
            context = tempa ## 18(#instance) x batchsize x 64
            mask = None
        else:
            if isinstance(prior, tuple) is True:
                prior, mask = prior
            else:
                mask = None 

            if prior.type() == 'torch.cuda.HalfTensor':
                self.down_proj_prior.half()
            context = self.down_proj_prior(prior)

            if prior_2 is not None:
                if prior_2.type() == 'torch.cuda.HalfTensor':
                    self.down_proj_prior_2.half()
                context2 = self.down_proj_prior_2(prior_2)
                context = torch.cat((context, context2), dim=0)

        tempa = self.non_linear_func(tempa)
        if len(tempa.shape) ==2:
            tempa = tempa.unsqueeze(1) ## 197 x batchsize x 64

        for z, layer in enumerate(self.mhsa_layers):
            if tempa.type() == 'torch.cuda.HalfTensor':
                layer.half()
                self.up_proj_mem.half()
            tempa = layer(tempa, context, tgt_mask=None,
                        memory_mask=None,
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=mask,
                        pos=None, query_pos=None)
        # pdb.set_trace()
        up = self.up_proj_mem(tempa)
        # pdb.set_trace()
        if len(up.shape) > len(x.shape):
            output = (up * self.scale).squeeze(1) + x
        else:
            output = (up * self.scale) + x
        return output

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.N_CTX # cfg.TRAINER.COOP.N_CTX
        ctx_init = args.CTX_INIT ## cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if args.CSC: # cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        # classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.CLASS_TOKEN_POSITION  # cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        # self.prompt_learner = PromptLearner(args, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        # self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)
        if len(x) > 1:
            x = x[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        raise ValueError
        return logits

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        """
        Focal Loss 实现
        :param gamma: 控制容易样本的损失缩小程度（常用 γ=2）
        :param alpha: 类别权重因子（适用于类别不平衡，常用 α=0.25）
        :param reduction: 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        计算 Focal Loss
        :param logits: 模型的 raw logits（未经过 sigmoid）
        :param targets: 真实标签（0 或 1）
        """
        # 计算 Sigmoid 后的概率
        # probs = torch.sigmoid(logits)
        probs = logits.clamp(min=1e-6, max=1.0 - 1e-6)  # 避免数值稳定性问题

        # 计算 p_t
        pt = probs * targets + (1 - probs) * (1 - targets)

        # 计算 Focal Loss
        ce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module 
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action/interaction classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """
    def __init__(self,
        args,
        detector: nn.Module,
        postprocessor: nn.Module,
        model: nn.Module,
        origin_text_embeddings: torch.tensor,
        object_embedding: torch.tensor,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15,
        object_class_to_target_class: List[list] = None,
        object_n_verb_to_interaction: List[list] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = model
        self.origin_text_embeddings = origin_text_embeddings
        self.object_embedding = object_embedding
        self.fix_mem = args.fix_mem
        self.visual_output_dim = model.image_encoder.output_dim
        self.visual_patchsize = model.image_encoder.patch_size
        self.visual_input_resolution = model.image_encoder.input_resolution
        self.object_n_verb_to_interaction = np.asarray(
                                object_n_verb_to_interaction, dtype=float
                            )

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class
        self.num_anno = kwargs["num_anno"]

        self.num_classes = num_classes
        self.use_multi_hot = args.use_multi_hot
           
        feat_dim = args.feat_dim
        self.feature = []
        if 'HO' in args.logits_type:
            self.feature.append('hum_obj')

        self.feature = '_'.join(self.feature)
        self.logits_type = args.logits_type

        num_shot = args.num_shot
        file1 = args.file1
        self.file1 = file1


        if args.zs:
            self.zs_type = args.zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]
        else:
            self.filtered_hoi_idx = []
            self.zs_type = None

        self.label_choice = args.label_choice
        self.mem_adapter = Adapter(feat_dim, prior_dim = feat_dim)

        if args.img_align is True:
            self.img_adapter = Adapter(2*feat_dim, mem_adpt_self = True, prior_dim = feat_dim)

        self.hoicls_txt = kwargs['hoicls_txt']
        self.self_adapt = args.self_adapt
        self.wo_sparsity = args.wo_sparsity
        if self.self_adapt == True:
            self.self_adapter = Adapter(feat_dim, mem_adpt_self = False, SA_only=False)

        self.wo_unseen_pred = args.wo_unseen_pred
        if 'HO' in self.logits_type:
            self.cache_model_HO, self.one_hots_HO, self.sample_lens_HO = self.load_cache_model(file1=file1, feature='hum_obj',num_classes=self.num_classes, num_shot=num_shot, filtered_hoi_idx = self.filtered_hoi_idx, 
                                                                                               use_multi_hot=self.use_multi_hot, label_choice=self.label_choice, num_anno=self.num_anno,
                                                                                               hoicls_txt = kwargs['hoicls_txt'], args = args)
            if self.wo_unseen_pred is True:
                self.sample_lens_HO = torch.sum(self.one_hots_HO, dim=0)
                self.seen_cls_list = torch.tensor([i for i in range(len(self.hoicls_txt)) if i not in self.filtered_hoi_idx])
            self.cache_model_HO, self.one_hots_HO, self.sample_lens_HO = self.cache_model_HO.cuda().float(), self.one_hots_HO.cuda().float(), self.sample_lens_HO.cuda().float()
        
        self.semloss = args.semloss
        ### basis features
        self.basis_feat_enable = args.basis_feat_enable
        if self.basis_feat_enable is True:
            self.basis_num = args.basis_num
            temp_hoicls_txt = kwargs['hoicls_txt'].T.numpy()
            if args.basis_feat_init == 'pca':
                n_components = args.recon_ratio_pca if args.recon_ratio_pca <= 1 else int(args.recon_ratio_pca)
                pca = PCA(n_components = n_components)
                pca.fit(temp_hoicls_txt)
                pca_hoicls_txt = pca.transform(temp_hoicls_txt)
                pca_hoicls_txt = torch.tensor(pca_hoicls_txt.T)

                pca_hoicls_txt = pca_hoicls_txt / pca_hoicls_txt.norm(dim=-1, keepdim=True)
                self.basis_feat = nn.Parameter(pca_hoicls_txt)
                hoicls_w = pca.components_
                self.hoicls_w = nn.Parameter(torch.tensor(hoicls_w).T)
                self.basis_num = len(pca_hoicls_txt)
                act_basis_num = self.basis_num // 2

                
            elif args.basis_feat_init == 'random':
                pca_hoicls_txt = torch.randn(self.basis_num, len(temp_hoicls_txt)).to(kwargs['hoicls_txt'].dtype)
                pca_hoicls_txt = pca_hoicls_txt / pca_hoicls_txt.norm(dim=-1, keepdim=True)
                self.basis_feat = nn.Parameter(pca_hoicls_txt)
                hoicls_w = kwargs['hoicls_txt'] @ torch.pinverse(pca_hoicls_txt)
                self.hoicls_w = nn.Parameter(hoicls_w)
                act_basis_num = self.basis_num // 2

            print("low rank: ", self.basis_num)

            self.recon_loss2 = nn.MSELoss(reduction='sum')
            self.ao_sep_basis = args.ao_sep_basis
            self.basis_feat_constraint = args.basis_feat_constraint
            self.kl_t = args.kl_t
            self.no_act_constraint = args.no_act_constraint
            self.HOI_train_w_b = args.HOI_train_w_b

            ### weight the action constraint according to unseen actions
            act_num_cls = len(torch.unique(torch.tensor([HOI_TO_AO[i][0] for i in range (len(self.hoicls_txt))]))) 
            act_num_cls_unseen = len(torch.unique(torch.tensor([HOI_TO_AO[i][0] for i in range (len(self.filtered_hoi_idx))]))) 
            ratio_uv = act_num_cls_unseen / act_num_cls 
            k = -torch.log(torch.tensor(1E-3)) / ratio_uv
            self.act_constraint_w_unseen = (1 - torch.exp(-k * ratio_uv))

            if self.ao_sep_basis is True:
                self.sep_frac = args.sep_frac
                
                if self.basis_feat_constraint != 'none':
                    actcls_txt = torch.randn(self.basis_num, len(temp_hoicls_txt)).to(kwargs['hoicls_txt'].dtype)
                    # import pdb
                    # pdb.set_trace()
                    actcls_txt = actcls_txt / actcls_txt.norm(dim=-1, keepdim=True)
                    self.act_related_index = torch.tensor(list(range(self.basis_num //self.sep_frac , self.basis_num)))
                    self.act_basis_feat = nn.Parameter(actcls_txt[self.act_related_index])
                    actcls_w = self.origin_text_embeddings @ torch.pinverse(actcls_txt[self.act_related_index])
                else:
                    actcls_txt = pca_hoicls_txt    
                    self.act_related_index = torch.tensor(list(range(self.basis_num //self.sep_frac , self.basis_num)))
                    actcls_w = self.origin_text_embeddings @ torch.pinverse(actcls_txt[self.act_related_index])
                    self.act_basis_feat = nn.Parameter(actcls_txt[self.act_related_index])
                self.actcls_w = nn.Parameter(actcls_w)
                self.act_related_index = nn.Parameter(self.act_related_index, requires_grad=False)
            

            
        self.individual_norm = True
        self.logits_type = args.logits_type #
        self.consist = True
        self.evaluate_type = 'detr' # gt, detr
        self.img_align = args.img_align
        self.use_type = 'crop'
        self.beta_cache = torch.tensor(10)
        self.alpha_cache = torch.tensor(1.0)
        self.semloss_weight = args.semloss_weight

        self.prior_type = args.prior_type
        self.finetune_adapter = True
        if self.prior_type == 'cbe':
            self.priors_initial_dim = self.visual_output_dim+5
        elif self.prior_type == 'cb':
            self.priors_initial_dim = 5
        elif self.prior_type == 'ce':
            self.priors_initial_dim = self.visual_output_dim+1
        elif self.prior_type == 'be':
            self.priors_initial_dim = self.visual_output_dim+4
        elif self.prior_type == 'c':
            self.priors_initial_dim = 1
        elif self.prior_type == 'b':
            self.priors_initial_dim = 4
        elif self.prior_type == 'e':
            self.priors_initial_dim = self.visual_output_dim
        else:
            raise NotImplementedError

        self.use_weight_pred = args.use_weight_pred
        if self.finetune_adapter:
            if 'HO' in self.logits_type:

                self.label_HO = nn.Parameter(self.one_hots_HO, requires_grad=False)
                if not self.use_weight_pred:
                    self.logit_scale_HO = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

            if 'U' in self.logits_type:
                self.label_U = nn.Parameter(self.one_hots_HO, requires_grad=False)
                if not self.use_weight_pred:
                    self.logit_scale_U = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

            if 'T' in self.logits_type:
                self.adapter_union_weight = nn.Parameter(self.origin_text_embeddings.clone().detach())
                if not self.use_weight_pred:
                    self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 
        
        if args.use_insadapter:
            if args.prior_method == 0:
                self.priors_downproj = MLP(self.priors_initial_dim, 128, 64, 3) # old 512+5   
            elif args.prior_method == 1:
                self.priors_downproj = MLP(self.priors_initial_dim * 2, 128, 64, 3) # old 512+5   
            elif args.prior_method == 2:
                self.learnable_prior = nn.Parameter(torch.empty(args.vis_prompt_num, 64))
                nn.init.xavier_normal_(self.learnable_prior)

        self.no_interaction_indexes = [9, 23, 30, 45, 53, 64, 75, 85, 91, 95, 106, 110, 128, 145, 159, 169, 173, 185, 193, 197, 207, 213, 223, 231, 234, 238, 242, 246, 251, 256, 263, 272, 282, 289, 294, 304, 312, 324, 329, 335, 341, 347, 351, 355, 362, 367, 375, 382, 388, 392, 396, 406, 413, 417, 428, 433, 437, 444, 448, 452, 462, 473, 482, 487, 501, 505, 515, 527, 532, 537, 545, 549, 557, 561, 566, 575, 583, 587, 594, 599]
        self.HOI_IDX_TO_OBJ_IDX = [
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14,
                14, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 39,
                39, 39, 39, 39, 39, 39, 39, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 56, 56, 56, 56,
                56, 56, 57, 57, 57, 57, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 60, 60,
                60, 60, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58,
                58, 58, 58, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 6, 6, 6, 6, 6,
                6, 6, 6, 62, 62, 62, 62, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 24, 24,
                24, 24, 24, 24, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 34, 34, 34, 34, 34,
                34, 34, 34, 35, 35, 35, 21, 21, 21, 21, 59, 59, 59, 59, 13, 13, 13, 13, 73,
                73, 73, 73, 73, 45, 45, 45, 45, 45, 50, 50, 50, 50, 50, 50, 50, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 67, 67, 67,
                67, 67, 67, 67, 74, 74, 74, 74, 74, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                54, 54, 54, 54, 54, 54, 54, 54, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                20, 10, 10, 10, 10, 10, 42, 42, 42, 42, 42, 42, 29, 29, 29, 29, 29, 29, 23,
                23, 23, 23, 23, 23, 78, 78, 78, 78, 26, 26, 26, 26, 52, 52, 52, 52, 52, 52,
                52, 66, 66, 66, 66, 66, 33, 33, 33, 33, 33, 33, 33, 33, 43, 43, 43, 43, 43,
                43, 43, 63, 63, 63, 63, 63, 63, 68, 68, 68, 68, 64, 64, 64, 64, 49, 49, 49,
                49, 49, 49, 49, 49, 49, 49, 69, 69, 69, 69, 69, 69, 69, 12, 12, 12, 12, 53,
                53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 72, 72, 72, 72, 72, 65, 65, 65, 65,
                48, 48, 48, 48, 48, 48, 48, 76, 76, 76, 76, 71, 71, 71, 71, 36, 36, 36, 36,
                36, 36, 36, 36, 36, 36, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 44, 44, 44, 44, 44, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 11, 11, 11, 11, 28, 28, 28, 28, 28, 28, 28, 28,
                28, 28, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 77, 77, 77, 77, 77,
                38, 38, 38, 38, 38, 27, 27, 27, 27, 27, 27, 27, 27, 70, 70, 70, 70, 61, 61,
                61, 61, 61, 61, 61, 61, 79, 79, 79, 79, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 25, 25, 25, 25, 25, 25, 25, 25, 75, 75, 75, 75, 40, 40, 40, 40, 40,
                40, 40, 22, 22, 22, 22, 22
            ]
        self.obj_to_no_interaction = torch.as_tensor([169, 23, 75, 159, 9, 64, 193, 575, 45, 566, 329, 505, 417, 246,
                                                        30,  85, 128, 145, 185, 106, 324, 238, 599, 347, 213, 583, 355, 545,
                                                        515, 341, 473, 482, 501, 375, 231, 234, 462, 527, 537,  53, 594, 304,
                                                        335, 382, 487, 256, 223, 207, 444, 406, 263, 282, 362, 428, 312, 272,
                                                        91,  95, 173, 242, 110, 557, 197, 388, 396, 437, 367, 289, 392, 413,
                                                        549, 452, 433, 251, 294, 587, 448, 532, 351, 561])

        self.epoch = 0
        self.COCO_CLASSES = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', \
                    'fire hydrant','N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',\
                    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', \
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', \
                    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', \
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', \
                    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', \
                    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.reserve_indices = [idx for (idx, name) in enumerate(self.COCO_CLASSES) if name != 'N/A']
        self.reserve_indices = self.reserve_indices + [91]
        self.reserve_indices = torch.as_tensor(self.reserve_indices)
        self.dataset = args.dataset
        self.hyper_lambda = args.hyper_lambda

        self.featmap_dropout = nn.Dropout(0.2)
        self.feat_mask_type = args.feat_mask_type
        self.use_insadapter = args.use_insadapter
        self.prior_method = args.prior_method
        self.box_proj = args.box_proj
        if self.box_proj:
            self.box_proj_mlp = MLP(8, 128, self.visual_output_dim, num_layers=3)
        if self.use_weight_pred:
            num_branch = len(self.logits_type.split('+'))
            self.weight_pred = Weight_Pred(input_dim=self.visual_output_dim*3, output_dim=num_branch)

        self.vcoco = True if args.dataset == 'vcoco' else False
        if args.dataset == 'vcoco':
            self.vcoco_rela = RelationNet()


        if 'U' in self.logits_type:
            self.vis_fuse = nn.Sequential(
                nn.Linear(self.visual_output_dim * 2, self.visual_output_dim),
                nn.ReLU(),
                nn.Linear(self.visual_output_dim, self.visual_output_dim),
                nn.ReLU(),
            )
        self.norm_pred = args.norm_pred
        self.seperate_ho = args.seperate_ho
        if self.seperate_ho != 0:
            self.ho_fuse = Adapter(feat_dim*2, mem_adpt_self = True, SA_only=False)
        self.unique_basis_weights = args.unique_basis_weights
        self.disentangle_basis = args.disentangle_basis
        self.fix_act_w = args.fix_act_w

  
        self.ho_pair_pt = args.ho_pair_pt
        self.pred_type = args.pred_type
        self.ho_pair_prior = args.ho_pair_prior
        self.pred_type = args.pred_type
        self.pt_init = args.pt_init
        # self.pt_attn = args.pt_attn
        if args.ho_pair_pt is True:
            vis_dim = 768 if feat_dim == 512 else 1024
            self.HO_pair_prior_text_embeddings = kwargs['HO_pair_text_embeddings']

            if self.ho_pair_prior == 1:
                self.pt_adapter = Adapter(256, mem_adpt_self = False, SA_only=False, prior_dim=feat_dim)
            elif self.ho_pair_prior == 2:
                self.pt_adapter = Adapter(256, mem_adpt_self = True)

            
            self.pt_spacial_proj = nn.Linear(256, vis_dim)
            self.pt_spacial_proj_norm = LayerNorm(vis_dim)


            if self.pt_init == 'pos+detr+fus':
                self.pt_token_fus = Adapter(256, mem_adpt_self = False, SA_only=False, prior_dim=36)
            else:
                self.pt_posenc_head = nn.Sequential(
                    nn.Linear(36, 128), nn.ReLU(),
                    nn.Linear(128, 256), nn.ReLU(),
                )            



    def positional_encoding(self, seq_len, d_model):
        """ 
        seq_len: length of sequence
        d_model: hidden layer dimension
        """
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        positional_embedding = torch.zeros((1, seq_len, d_model))

        div_term = torch.pow(10000.0, 2*torch.arange(0, d_model//2)/d_model) 
        positional_embedding[0, :, 0::2] = torch.sin(pos / div_term)
        positional_embedding[0, :, 1::2] = torch.cos(pos / div_term)
        
        """ OR
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        positional_embedding[0, :, 0::2] = torch.sin(pos * div_term)
        positional_embedding[0, :, 1::2] = torch.cos(pos * div_term)
        """
        return positional_embedding

    def load_cache_model(self,file1, feature='uni',num_classes=117, num_shot=10, filtered_hoi_idx=[], use_multi_hot=False, label_choice='random', 
                         num_anno=None, hoicls_txt = None, args = None):  ## √
        

        if args is not None and args.dataset =='vcoco':
            HOI_TO_AO_hoi = HOI_TO_AO_COCO
            numcls = 236
            cache_labels = np.array([HOI_TO_AO_COCO[i][0] for i in range(numcls)])
        else:
            HOI_TO_AO_hoi = HOI_TO_AO
            numcls = 600
            cache_labels = np.array(HOI_IDX_TO_ACT_IDX)

        cache_models = hoicls_txt / hoicls_txt.norm(dim=-1, keepdim=True)

        labels = torch.tensor(cache_labels)
        labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        
        return cache_models, labels, torch.sum(labels, dim=0)

    
    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:  ### √

        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)
        
        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)
        if self.dataset == 'swig':
            prior_h = s_h.unsqueeze(-1).repeat(1, self.num_classes)
            prior_o = s_o.unsqueeze(-1).repeat(1, self.num_classes)
            return torch.stack([prior_h, prior_o])
        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping 
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])
    

    def compute_roi_embeddings(self, features: OrderedDict, image_size: Tensor, region_props: List[dict], targets = None, fix_mem = False, vcoco = False, images = None, paired_tokens=None):
        device = features.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        attn_maps_collated = []
        all_logits = []

        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        gt_feats_collated = []
        pair_feats_collated = []
        gt_all_logits = []
        pair_logits = []
        pair_prior = []
        gt_labels = []
        glb_feat_list = []
        adapter_feat_list = []
        ho_adapter_feat_list = []
 

        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx]

            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]
            # Skip image when there are no valid human-object pairs
            if n_h == 0 or n <= 1:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten(); y = y.flatten()

            # extract single roi features
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1)
            # union_boxes_list.append(union_boxes)
           
            spatial_scale = 1 / (image_size[0,0]/local_features.shape[1])
            union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)
            single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)


            if self.feat_mask_type == 0:
                union_features = self.featmap_dropout(union_features).flatten(2).mean(-1)
                single_features = self.featmap_dropout(single_features).flatten(2).mean(-1)
            elif self.feat_mask_type == 1:
                union_features = union_features.flatten(2).mean(-1)
                single_features = single_features.flatten(2).mean(-1)
            human_features = single_features[x_keep]
            object_features = single_features[y_keep]

            if self.individual_norm: ## todo should use norm during finetuning?? 
                concat_feat_original = torch.cat([human_features,object_features, union_features],dim=-1)
                human_features = human_features / human_features.norm(dim=-1, keepdim=True)
                object_features = object_features / object_features.norm(dim=-1, keepdim=True)
                union_features = union_features / union_features.norm(dim=-1, keepdim=True)
                if self.feature == 'hum_obj_uni':
                    concat_feat = torch.cat([human_features, object_features, union_features],dim=-1) 
                elif self.feature == 'hum_obj':
                    concat_feat = torch.cat([human_features, object_features], dim=-1)
                elif self.feature == 'uni':
                    concat_feat = union_features
            else:
                concat_feat = torch.cat([human_features,object_features, union_features],dim=-1) 
                concat_feat = concat_feat/concat_feat.norm(dim=-1,keepdim=True) 

            if self.basis_feat_enable is True:
                if self.HOI_train_w_b == 'w':
                    # temp_basis = self.basis_feat.clone().detach()
                    txt_optimized = self.hoicls_w @ self.basis_feat.clone().detach()
                elif self.HOI_train_w_b == 'b':
                    txt_optimized = self.hoicls_w.clone().detach() @ self.basis_feat
                elif self.HOI_train_w_b == 'both':
                    txt_optimized = self.hoicls_w @ self.basis_feat
                else:
                    txt_optimized = self.hoicls_w.clone().detach() @ self.basis_feat.clone().detach()
            else:
                txt_optimized = self.cache_model_HO
            if self.self_adapt == True and self.fix_mem is False:
                txt_optimized = self.self_adapter(txt_optimized.unsqueeze(0), 
                                                        txt_optimized.unsqueeze(0)).squeeze(0)

            if vcoco is True:
                adapter_feat = (self.mem_adapter(txt_optimized.unsqueeze(1), 
                                local_features.flatten(1).mean(-1).unsqueeze(0).unsqueeze(1).repeat(len(self.hoicls_txt),1,1))).squeeze(1)
            else:
                if self.fix_mem is True:
                    # adapter_feat = self.attri_weights_to_hoicls @ (self.attri_feats)
                    adapter_feat = txt_optimized
                else:
                    adapter_feat = (self.mem_adapter(txt_optimized.unsqueeze(1),
                                local_features.flatten(1).mean(-1).unsqueeze(0).unsqueeze(1).repeat(len(self.hoicls_txt),1,1))).squeeze(1)

            if len(adapter_feat.shape) == 3:
                adapter_feat = adapter_feat.squeeze()

            if self.img_align is True:
                vis_adapter_feat = self.img_adapter(torch.cat([human_features, object_features], dim=-1).unsqueeze(0))    #.squeeze(0)
                vis_adapter_feat = vis_adapter_feat.squeeze(0)
            else:
                vis_adapter_feat = torch.cat([human_features, object_features], dim=-1)
            
            ### union vis hum and obj
            if self.seperate_ho == 1:
                if self.basis_feat_enable is True:
                    if self.basis_feat_constraint == 'none':
                        if self.HOI_train_w_b == 'w':
                            act_txt_feat = self.actcls_w @  self.basis_feat[self.act_related_index].clone().detach()
                        elif self.HOI_train_w_b == 'b':
                            act_txt_feat = self.actcls_w.clone().detach() @  self.basis_feat[self.act_related_index]
                        elif self.HOI_train_w_b == 'both':
                            act_txt_feat = self.actcls_w @  self.basis_feat[self.act_related_index]
                        else:
                            act_txt_feat = self.actcls_w.clone().detach() @  self.basis_feat[self.act_related_index].clone().detach()
                    else:
                        temp_act_basis_feat = self.act_basis_feat.clone().detach()             
                        act_txt_feat = self.actcls_w @ temp_act_basis_feat
                else:
                    act_txt_feat = self.origin_text_embeddings.to(adapter_feat.device)
                
                if self.dataset == 'hicodet':
                    act_txt_feat = act_txt_feat[HOI_IDX_TO_ACT_IDX]
                    act_txt_feat = act_txt_feat / act_txt_feat.norm(dim=-1, keepdim=True)
                    obj_txt_feat = self.object_embedding[HOI_IDX_TO_OBJ_IDX].to(adapter_feat.device)
                else:
                    act_txt_feat = act_txt_feat[[HOI_TO_AO_COCO[i][0] for i in range(236)]]
                    act_txt_feat = act_txt_feat / act_txt_feat.norm(dim=-1, keepdim=True)
                    obj_txt_feat = self.object_embedding[[HOI_TO_AO_COCO[i][1] for i in range(236)]].to(adapter_feat.device)

                ho_adapt_feat = self.ho_fuse(torch.cat((act_txt_feat, obj_txt_feat), dim=-1).unsqueeze(0)).squeeze(0)
                ho_adapter_feat_list.append(ho_adapt_feat)

            else:
                union_humobj_vis = self.vis_fuse(vis_adapter_feat )

            logits = None
            if 'ho' in self.pred_type:
                if self.seperate_ho != 0:
                    img_feat_HO = vis_adapter_feat
                    phi_union_HO =  (vis_adapter_feat @ (ho_adapt_feat).T)
                else:
                    img_feat_HO = union_humobj_vis
                    phi_union_HO =  (union_humobj_vis @ adapter_feat.T)
                if self.wo_unseen_pred is True and self.training is True:
                    phi_union_HO = phi_union_HO[:, self.seen_cls_list]
                    label_HO = self.label_HO[self.seen_cls_list]
                else:
                    label_HO = self.label_HO
                logits_cache_HO = ((phi_union_HO @ label_HO) / self.sample_lens_HO) /2
 
                logits = logits_cache_HO * self.logit_scale_HO if logits is None else logits + logits_cache_HO * self.logit_scale_HO

            if 'u'in self.pred_type and 'l' in self.pred_type: 
                logit_scale_U = self.logit_scale_U / 2
            else: 
                logit_scale_U = self.logit_scale_U


            if 'u' in self.pred_type:
                phi_union_U = union_features @ adapter_feat.squeeze(1).T
                if self.wo_unseen_pred is True and self.training is True:
                    phi_union_U = phi_union_U[:, self.seen_cls_list]
                    label_HO = self.label_HO[self.seen_cls_list]
                else:
                    label_HO = self.label_HO
                logits_cache_U = (phi_union_U @ label_HO) / self.sample_lens_HO

                logits = logits_cache_U * logit_scale_U if logits is None else logits + logits_cache_U * logit_scale_U

            if 'l' in self.pred_type:
                phi_union_L = paired_tokens[b_idx][:len(union_features)] @ adapter_feat.squeeze(1).T
                if self.wo_unseen_pred is True and self.training is True:
                    phi_union_L = phi_union_L[:, self.seen_cls_list]
                    label_HO = self.label_HO[self.seen_cls_list]
                else:
                    label_HO = self.label_HO
                logits_cache_L = (phi_union_L @ label_HO) / self.sample_lens_HO
              
                logits = logits_cache_L * logit_scale_U if logits is None else logits + logits_cache_L * logit_scale_U

            ### prior score
            pr_i = self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
         

            all_logits.append(logits)
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(pr_i)
            
            glb_feat = local_features.flatten(1).mean(-1)
            glb_feat_list.append(glb_feat)
            adapter_feat_list.append(adapter_feat)
        
        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, gt_feats_collated, pair_feats_collated, glb_feat_list, adapter_feat_list, ho_adapter_feat_list
          
    def recover_boxes(self, boxes, size):  
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets): ## for training
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        
        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)
        # print("pair gt,",len(x),len(y))
        # IndexError: tensors used as indices must be long, byte or bool tensors
        if self.dataset == 'swig' and self.training:
            if len(y) > 0:
                tgthoi_y = torch.as_tensor([self.unique_hois[origin_hoi_idx.item()] for origin_hoi_idx in targets['hoi'][y]], device=boxes_h.device)
                labels[x, tgthoi_y] = 1
        elif self.num_classes == 117 or self.num_classes == 24 or self.num_classes == 407:
            labels[x, targets['labels'][y]] = 1  ## target['labels']: verb/action
        else:
            labels[x, targets['hoi'][y]] = 1
        # print("#(labels==1) = ", torch.sum(labels))
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets, gt_feats, pair_feats,): ### loss
        ## bx, bo: indices of boxes
        temp_labels = [
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ]
        labels = torch.cat(temp_labels)

        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        num_one_label = torch.sum(labels)
        logits = torch.cat(logits) 
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]


        
        n_p = len(torch.nonzero(labels))

        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier() 
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        
        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='none',
            alpha=self.alpha, gamma=self.gamma
            )

        if n_p >=1:
            return loss.sum() / n_p, temp_labels
        else:
            return loss.sum()*n_p, temp_labels

    def prepare_region_proposals(self, results, hidden_state=None): ## √ detr extracts the human-object pairs
        region_props = []
        for bz_i, res in enumerate(results):
            # if faster_rcnn == False:
            sc, lb, bx = res.values()
            # else:
            #     bx, sc, lb = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            if hidden_state is not None:
                hs = hidden_state[bz_i]
                hs=hs[keep].view(-1, 256)
            
            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])
            if hidden_state is None:
                region_props.append(dict(
                    boxes=bx[keep],
                    scores=sc[keep],
                    labels=lb[keep],
                ))
            else:
                region_props.append(dict(
                    boxes=bx[keep],
                    scores=sc[keep],
                    labels=lb[keep],
                    hidden_state=hs[keep]
                ))


        return region_props

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes, flag = 0): ### √
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size,  in zip(
            boxes, bh, bo, logits, prior, objects, image_sizes,
        ):

            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            # if flag == 1:
            #     pdb.set_trace()      
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections


    def get_prior(self, region_props, image_size, prior_method): ##  for adapter module training
        max_feat = self.priors_initial_dim
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        mask = torch.ones((len(region_props),max_length),dtype=torch.bool,device=region_props[0]['boxes'].device)
        priors = torch.zeros((len(region_props),max_length, max_feat), dtype=torch.float32, device=region_props[0]['boxes'].device)
        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        
        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] / scale_fct[b_idx][None,:]
            scores = props['scores']
            labels = props['labels']
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1:
                print(n_h,n)

            
            object_embs = self.object_embedding[labels.to(self.object_embedding.device)]

            mask[b_idx,:n] = False
            
            if self.prior_type == 'cbe':
                priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
                priors[b_idx,:n,5:self.visual_output_dim+5] = object_embs
        
            elif self.prior_type == 'cb':
                priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
            elif self.prior_type == 'ce':
                priors[b_idx,:n,:1] = scores.unsqueeze(-1)
                priors[b_idx,:n,1:self.visual_output_dim+1] = object_embs
            elif self.prior_type == 'be':
                priors[b_idx,:n,:4] = boxes
                priors[b_idx,:n,4:self.visual_output_dim+4] = object_embs
            elif self.prior_type == 'c':
                priors[b_idx,:n,:1] = scores.unsqueeze(-1)
            elif self.prior_type == 'b':
                priors[b_idx,:n,:4] = boxes
            elif self.prior_type == 'e':
                priors[b_idx,:n,:self.visual_output_dim] = object_embs
            else:
                raise NotImplementedError

        if prior_method == 0:
            priors = self.priors_downproj(priors)
        elif prior_method == 1:
            pair_wise_priors = []
            for b_idx, props in enumerate(region_props):
                boxes = props['boxes'] / scale_fct[b_idx][None,:]
                scores = props['scores']
                labels = props['labels']
                is_human = labels == self.human_idx
                n_h = torch.sum(is_human); n = len(boxes)
                if n_h == 0 or n <= 1:
                    pair_wise_priors.append(torch.zeros(0, 0), )
                    print(n_h,n)
                    continue
                instance_wise_prior = priors[b_idx, :n]
                # Get the pairwise indices
                x, y = torch.meshgrid(
                    torch.arange(n, device=instance_wise_prior.device),
                    torch.arange(n, device=instance_wise_prior.device)
                )
                # Valid human-object pairs
                x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
                if len(x_keep) == 0:
                    # Should never happen, just to be safe
                    raise ValueError("There are no valid human-object pairs")
                
                # extract single roi features
                sub_prior = instance_wise_prior[x_keep]
                obj_prior = instance_wise_prior[y_keep]
                
                pair_wise_priors.append(torch.cat((sub_prior, obj_prior), dim=-1))
            
            max_length = max(p.shape[0] for p in pair_wise_priors)
            mask = torch.ones((len(region_props),max_length),dtype=torch.bool,device=region_props[0]['boxes'].device)
            priors = torch.zeros((len(region_props),max_length, max_feat*2), dtype=torch.float32, device=region_props[0]['boxes'].device)
            for b_idx, props in enumerate(region_props):
                num_pair = pair_wise_priors[b_idx].shape[0]
                if num_pair > 0:
                    mask[b_idx, :num_pair] = False
                    priors[b_idx, :num_pair] = pair_wise_priors[b_idx]
            priors = self.priors_downproj(priors)   
        elif prior_method == 2:
            priors = self.learnable_prior.unsqueeze(0).repeat(len(region_props), 1, 1)
            mask = torch.zeros((priors.shape[0], priors.shape[1]), dtype=torch.bool,device=region_props[0]['boxes'].device)

        return (priors, mask)
    
    def get_pair_prior(self, region_props, image_size, clip_img=None, drawmask=None): 
        paired_priors_orig = []
        HOI_tokens = []
        paired_boxes = []
        for i, rp in enumerate(region_props):
            boxes, scores, labels, embeds = rp.values()
            n_h = self.check_human_instances(labels)
            n = len(boxes)
            x, y = torch.meshgrid(
                    torch.arange(n, device=boxes.device),
                    torch.arange(n, device=boxes.device)
                    )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)

            ### generate paired boxes
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            paired_boxes.append((sub_boxes, obj_boxes))


            pairwise_spatial = compute_spatial_encodings(
            [boxes[x_keep],], [boxes[y_keep],], [image_size[i],]
            )   

            obji_HO_pair_text_embeddings = [self.HO_pair_prior_text_embeddings[i].to(pairwise_spatial.device) for i in labels[y_keep].tolist()]
            object_embs = self.object_embedding.to(pairwise_spatial.device)[torch.tensor(labels[y_keep])]
            
            if self.pt_init == 'pos+detr':
                spatial_embeddings = self.pt_posenc_head(pairwise_spatial)
                HOI_token = (embeds[x_keep] + embeds[y_keep])/2 + spatial_embeddings
            elif self.pt_init == 'pos':
                spatial_embeddings = self.pt_posenc_head(pairwise_spatial)
                HOI_token = spatial_embeddings
            elif self.pt_init == 'detr':
                HOI_token = (embeds[x_keep] + embeds[y_keep])/2
            elif self.pt_init == 'pos+detr+fus':
                HOI_token = self.pt_token_fus((embeds[x_keep]+ embeds[y_keep]).unsqueeze(0)/2, pairwise_spatial.unsqueeze(0)).squeeze(0)
            HOI_tokens.append(HOI_token)

            paired_priors_orig.append([torch.cat((pr_txt_i, object_embs[idx].unsqueeze(0))) for idx, pr_txt_i in enumerate(obji_HO_pair_text_embeddings)]) ### batch * HO-num * prior-num * dim

        pt_adapter_feats = []
        interact_feat_list = []
        for b_idx, pr_i in enumerate(paired_priors_orig):
            if len(pr_i) == 0:
                pt_adapter_feat = self.pt_spacial_proj(HOI_tokens[b_idx])
                pt_adapter_feat = self.pt_spacial_proj_norm(pt_adapter_feat)

                pt_adapter_feats.append(pt_adapter_feat)
                continue

            max_length = max(len(pr_i_obji) for pr_i_obji in pr_i)

            mask = torch.ones((len(pr_i),max_length),dtype=torch.bool,device="cuda")
            priors = torch.zeros((len(pr_i),max_length, object_embs.shape[-1]), dtype=torch.float32, device="cuda")
    
            for obji, pr_i_obji in enumerate(pr_i):
                mask[obji, :len(pr_i_obji)] = False
                priors[obji, :len(pr_i_obji)] = pr_i_obji

            if self.ho_pair_prior == 1:
                pt_adapter_feat = self.pt_adapter(HOI_tokens[b_idx].unsqueeze(0), (priors.permute(1,0,2), mask)).squeeze(0) 
            elif self.ho_pair_prior == 2:
                pt_adapter_feat = self.pt_adapter(HOI_tokens[b_idx].unsqueeze(0)).squeeze(0) 
            else:
                pt_adapter_feat = HOI_tokens[b_idx]

            pt_adapter_feat = self.pt_spacial_proj(pt_adapter_feat)
            pt_adapter_feat = self.pt_spacial_proj_norm(pt_adapter_feat)


            pt_adapter_feats.append(pt_adapter_feat)
            
    
        return pt_adapter_feats, interact_feat_list


    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        if not torch.all(labels[:n_h]==self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return n_h

    def prepare_target_hois(self, targets, device):
        unique_hois, cnt = {}, 0
        tgt_ids = []
        for t in targets:
            for hoi in t["hoi"]:
                hoi_id = hoi.item()
                if self.training:
                    # Only consider the texts within each mini-batch
                    if hoi_id not in unique_hois:
                        unique_hois[hoi_id] = cnt
                        cnt += 1
                    tgt_ids.append(unique_hois[hoi_id])
                else:
                    # Consider all hois in the dataset
                    tgt_ids.append(hoi_id)
        tgt_ids = torch.as_tensor(tgt_ids, dtype=torch.int64, device=device)
        return unique_hois


    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if not self.finetune_adapter:
            raise NotImplementedError
            if self.training and targets is None:
                raise ValueError("In training mode, targets should be passed")
            image_sizes = torch.as_tensor([
                im.size()[-2:] for im in images
            ], device=images[0].device)
            region_props = self.get_region_proposals(targets)  # exhaustively generate the human-object pairs from the detr results
            feat_local_old = self.clip_model.encode_image(images[0])
            feat_local = feat_local_old[:,1:,:].transpose(1,2).view(feat_local_old.shape[0],-1, 7, 7).float()
            cls_feature = feat_local_old[:,0,:]
            # use the gt crop
            
            if self.evaluate_type == 'gt':
                if self.use_type == 'crop':
                    logits, prior, bh, bo, objects, boxes = self.compute_roi_embeddings_targets(cls_feature.unsqueeze(0), image_sizes, targets)
                else: #### ignore 
                    logits, prior, bh, bo, objects = self.compute_roi_embeddings(feat_local, image_sizes, region_props)
            elif self.evaluate_type == 'detr':      
                logits, prior, bh, bo, objects, = self.compute_crop_embeddings(cls_feature.unsqueeze(0), image_sizes, region_props, targets)
                boxes = [r['boxes'] for r in region_props]
            # boxes = [r['boxes'] for r in region_props]
            
            if self.training:
                interaction_loss, _ = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets, interactiveness)
                loss_dict = dict(
                    interaction_loss=interaction_loss
                )
                return loss_dict
            
            detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
            return detections
        else:
            if self.training and targets is None:
                raise ValueError("In training mode, targets should be passed")
            batch_size = len(images)
            images_orig = [im[0].float() for im in images]
            images_clip = [im[1] for im in images]

            device = images_clip[0].device
            image_sizes = torch.as_tensor([
                im.size()[-2:] for im in images_clip
            ], device=device)
            image_sizes_orig = torch.as_tensor([
                im.size()[-2:] for im in images_orig
                ], device=device)
            
            if self.dataset == 'swig':
                ## boxes should be xyxy in the origin whwh coordinate space
                outputs = self.detector(images_orig)
                detections = self.postprocessor(
                    outputs, image_sizes
                )
                results = detections
                
            else:

                # assert mask is not None2
                if isinstance(images_orig, (list, torch.Tensor)):
                    images_orig = nested_tensor_from_tensor_list(images_orig)
                features, pos = self.detector.backbone(images_orig)
                src, mask = features[-1].decompose()
                hs, detr_memory = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])
    
                outputs_class = self.detector.class_embed(hs) # 6x8x100x81 or 6x8x100x92
                outputs_coord = self.detector.bbox_embed(hs).sigmoid() # 6x8x100x4 

                if outputs_class.shape[-1] == 92:
                    outputs_class = outputs_class[:, :, :, self.reserve_indices]
                    assert outputs_class.shape[-1] == 81, 'reserved shape NOT match 81'

                results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
         
                results = self.postprocessor(results, image_sizes)
                region_props = self.prepare_region_proposals(results, hidden_state=hs[-1])

            if self.use_insadapter:
                priors = self.get_prior(region_props,image_sizes.to(region_props[0]['labels'].device), self.prior_method) ## priors: (prior_feat, mask): (batch_size*14*64, batch_size*14)
                if torch.isnan(priors[0]).any() == True:
                    pdb.set_trace()
            else: 
                priors = None

            images_clip = nested_tensor_from_tensor_list(images_clip)
            feat_new = None
            context = None  

            if self.ho_pair_pt is True:
                pt_adapter_feats, interact_feat_list = self.get_pair_prior(region_props, image_sizes.to(region_props[0]['labels'].device), clip_img = images_clip, drawmask=context)
            else:
                pt_adapter_feats = None
            

            feat_global, feat_local, paired_tokens = self.clip_head.image_encoder(images_clip.decompose()[0], priors, context=context, pair_prior = (pt_adapter_feats, None))   


            if self.dataset == 'swig' and self.training:
                self.unique_hois = self.prepare_target_hois(targets=targets, device=device)
                self.num_classes = len(self.unique_hois)
 
            logits, prior, bh, bo, objects, gt_feats, pair_feats, glb_feat, adapter_feat_list, ho_adapter_feat_list = self.compute_roi_embeddings(feat_local, image_sizes, region_props, targets = targets, fix_mem=self.fix_mem, vcoco = self.vcoco, images = images, paired_tokens=paired_tokens)
            gt_all_logits = None
            boxes = [r['boxes'] for r in region_props] 

            if self.training:
                interaction_loss, labels_matched = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets, gt_feats, pair_feats)

                if self.fix_mem is False or self.semloss is True:   ## self.txt_mem_init is False and 
              
                    annotation = pickle.load(open(self.file1,'rb'))
                    obj_embeddings = [[] for i in range(self.num_classes)]
                    hum_embeddings = [[] for i in range(self.num_classes)]
                    real_verbs = [[] for i in range(self.num_classes)]
                    hoi_clses = [[] for i in range(self.num_classes)]
                    filenames = list(annotation.keys())
                    # verbs_iou = [[] for i in range(self.num_classes)]
                    from hico_text_label import HOI_TO_AO
                    if 'COCO' in targets[0]['filename']:
                        # MAP_AO_TO_HOI = MAP_AO_TO_HOI_COCO
                        HOI_TO_AO = HOI_TO_AO_COCO
                    for tgti in range(len(targets)):
                        anno = annotation[targets[tgti]['filename']]
                        if self.num_classes == 117 or self.num_classes == 24: verbs = anno['verbs']
                        else: verbs = (self.object_n_verb_to_interaction[anno['objects'], anno['verbs']]).astype(int)
                        num_ho_pair = len(anno['boxes_h'])
                        anno['real_verbs'] = np.zeros(shape=(num_ho_pair, self.num_classes))
                        for i in range(num_ho_pair):
                            anno['real_verbs'][i][verbs[i]] = 1

                        similar_ho = box_iou(torch.as_tensor(anno['boxes_h']), torch.as_tensor(anno['boxes_h']))* \
                            box_iou(torch.as_tensor(anno['boxes_o']), torch.as_tensor(anno['boxes_o']))
                        similar_ho = similar_ho * (torch.ones_like(similar_ho)-torch.eye(len(similar_ho)))
                        similar_ho_idx = similar_ho > 0.6
                        
                        if len(verbs) == 0:
                            print(targets[tgti]['filename'])

                        delete_verb = []
                        for i, v in enumerate(verbs):
                            if 'hico' in self.file1: ## TODO ??? why vcoco list idx out of range
                                if self.num_classes == 117:
                                    # pdb.set_trace()
                                    if anno['verbs'][i] not in self.object_class_to_target_class[anno['objects'][i]]:
                                        continue
                                elif self.num_classes == 600:
                                    if v in self.filtered_hoi_idx:
                                        continue
                            if len(torch.nonzero(similar_ho_idx[i])) > 0:
                                if verbs[i] not in delete_verb:
                                    simi_verb = [verbs[i] for i in torch.nonzero(similar_ho_idx[i])[:,0]]
                                    simi_ind = torch.nonzero(similar_ho_idx[i])[:,0].tolist()
                                    simi_verb.append(verbs[i])
                                    simi_ind.append(i)
                                    rand_ind = random.choice(simi_ind)
                                    if rand_ind in delete_verb:
                                        continue
                                    v_append = verbs[rand_ind]
                                    for del_i in simi_ind:
                                        delete_verb.append(del_i)
                                    
                                    obj_embeddings[v_append].append(anno['object_features'][rand_ind] / np.linalg.norm(anno['object_features'][rand_ind]))
                                    hum_embeddings[v_append].append(anno['huamn_features'][rand_ind] / np.linalg.norm(anno['huamn_features'][rand_ind]))
                                    real_verbs[v_append].append(anno['real_verbs'][rand_ind])
                                    # pdb.set_trace()
                                    if 'HICO' in targets[tgti]['filename']:
                                        hoi_clses[v_append].append(MAP_AO_TO_HOI[anno['verbs'][rand_ind], anno['objects'][rand_ind]])
                                    else:
                                        hoi_clses[v_append].append(MAP_AO_TO_HOI_COCO[(anno['verbs'][rand_ind], anno['objects'][rand_ind])])
    
                            else:
                                obj_embeddings[v].append(anno['object_features'][i] / np.linalg.norm(anno['object_features'][i]))
                                hum_embeddings[v].append(anno['huamn_features'][i] / np.linalg.norm(anno['huamn_features'][i]))
                                real_verbs[v].append(anno['real_verbs'][i])
                                if 'HICO' in targets[tgti]['filename']:
                                    hoi_clses[v].append(MAP_AO_TO_HOI[anno['verbs'][i], anno['objects'][i]])
                                else:
                                    hoi_clses[v].append(MAP_AO_TO_HOI_COCO[(anno['verbs'][i], anno['objects'][i])])

                    cache_models_lst, each_lens_lst = [], []
                    real_verbs_lst = []
                    target_cls_list = []

                    indexes = np.arange(len(hum_embeddings))
                    count = -1
                    for i, hum_emb, obj_emb, real_v, hoi_c in (zip(indexes, hum_embeddings, obj_embeddings, real_verbs, hoi_clses)):
                        count += 1
                        hum_emb =  torch.as_tensor(np.array(hum_emb)).float()   
                        obj_emb = torch.as_tensor(np.array(obj_emb)).float()
                        real_v = torch.as_tensor(np.array(real_v))
                        new_embeddings = torch.cat([hum_emb, obj_emb], dim=-1)
                        new_embeddings = new_embeddings.cuda().float()
                        num_shot = 2
                        num_to_select = min(hum_emb.shape[0], num_shot)

                        if num_to_select < hum_emb.shape[0]:
                            topk_idx = torch.randperm(new_embeddings.shape[0])[:num_to_select] 
                            new_embeddings = new_embeddings[topk_idx]
                            real_v = real_v[topk_idx]
                            hoi_c = torch.as_tensor(hoi_c)[topk_idx]
                        else:
                            hoi_c = torch.as_tensor(hoi_c)
                        
                        cache_models_lst.append(new_embeddings)
                        each_lens_lst.append(num_to_select)
                        real_verbs_lst.append(real_v)
                        target_cls_list.append(hoi_c)
  

                    target_classes_label = torch.cat(target_cls_list, dim=0).type(torch.long).to(device)
                    unseen_list = [j for j in self.filtered_hoi_idx if HOI_TO_AO[j][1] in [HOI_TO_AO[i.item()][1] for i in target_classes_label]]
                    if self.dataset != 'vcoco':
                        select_semloss_index = -1

                        vis_seen = adapter_feat_list[select_semloss_index][target_classes_label]
                        visseen_similar = cal_similarity(vis_seen, vis_seen)
                        txtseen_similar = cal_similarity(self.hoicls_txt.to(device)[target_classes_label], 
                                                        self.hoicls_txt.to(device)[target_classes_label])

                        if self.seperate_ho != 0:
                            vis_seen2 = ho_adapter_feat_list[select_semloss_index][target_classes_label]
                            visseen_similar2 = cal_similarity(vis_seen2, vis_seen2)
                            relation_ho_cls_seen = (kl_loss(visseen_similar, txtseen_similar) + kl_loss(visseen_similar2, txtseen_similar) ) / 2
                        else:
                            relation_ho_cls_seen = kl_loss(visseen_similar, txtseen_similar) 

                        if self.zs_type is not None and len(unseen_list) > 0:
                            vis_unseen = adapter_feat_list[select_semloss_index][unseen_list]
                            visunseen_similar = cal_similarity(vis_unseen, vis_unseen)
                            txt_unseen = self.hoicls_txt.to(device)[unseen_list]
                            txtunseen_similar = cal_similarity(txt_unseen, txt_unseen)

                            visall_similar = cal_similarity(torch.cat((vis_seen, vis_unseen), dim=0),
                                                        torch.cat((vis_seen, vis_unseen), dim=0))
                            txtall_similar = cal_similarity(torch.cat((self.hoicls_txt.to(device)[target_classes_label], txt_unseen), dim=0),
                                                        torch.cat((self.hoicls_txt.to(device)[target_classes_label], txt_unseen), dim=0))


                            if self.seperate_ho != 0:
                                vis_unseen2 = ho_adapter_feat_list[select_semloss_index][unseen_list]
                                visunseen_similar2 = cal_similarity(vis_unseen2, vis_unseen2)
                                relation_ho_cls_unseen = (kl_loss(visunseen_similar, txtunseen_similar) + kl_loss(visunseen_similar2, txtunseen_similar) ) / 2

                                visall_similar2 = cal_similarity(torch.cat((vis_seen2, vis_unseen2), dim=0),
                                                        torch.cat((vis_seen2, vis_unseen2), dim=0))
                                relation_ho_cls_all = (kl_loss(visall_similar, txtall_similar) + kl_loss(visall_similar2, txtall_similar) ) / 2
                            else:
                                relation_ho_cls_unseen = kl_loss(visunseen_similar, txtunseen_similar)
                                relation_ho_cls_all = kl_loss(visall_similar, txtall_similar)


                            loss_dict = dict(
                                interaction_loss=interaction_loss,
                                sem_loss = (relation_ho_cls_seen+relation_ho_cls_unseen+relation_ho_cls_all)/3*self.semloss_weight,
                            )
                        else:
                            loss_dict = dict(
                                interaction_loss=interaction_loss,
                                sem_loss = (relation_ho_cls_seen)*self.semloss_weight,
                            )
                    else:
                        loss_dict = dict(interaction_loss=interaction_loss)
                else:
                    loss_dict = dict(interaction_loss=interaction_loss)

                if interaction_loss.isnan():
                    pdb.set_trace()

                if self.basis_feat_enable is True and self.fix_mem is False:
                    temp_hoicls_w = self.hoicls_w.clone().detach()
                    loss_dict['recon_loss2'] = self.recon_loss2(temp_hoicls_w @ self.basis_feat, self.hoicls_txt.to(self.basis_feat.device)) * 0.1
                    if self.wo_sparsity is False:
                        loss_dict['loss_w_sparse'] = (torch.abs(self.hoicls_w)).sum(-1).mean(0) * 0.1 
                    
                    if self.unique_basis_weights is True:
                        tempb = self.hoicls_w.T 
                        tempb = tempb / tempb.norm(dim=-1, keepdim=True)
                        tempc = tempb @ tempb.T
                        tempc = tempc * (torch.ones_like(tempc).to(tempc.device) - torch.eye(len(tempc)).to(tempc.device))
                        loss_dict['loss_w_unique'] = torch.abs(tempc).sum()*0.001

                    # tempe = self.hoicls_w.T.softmax(dim=-1)
                    # loss_dict['loss_w_entropy'] = -(tempe*torch.log(tempe)).sum(dim=-1).mean() * 0.3

                    if self.disentangle_basis is True:
                        tempa = self.basis_feat / self.basis_feat.norm(dim=-1, keepdim=True)
                        tempa = tempa @ tempa.T 
                        tempa = tempa * (torch.ones_like(tempa).to(tempa.device) - torch.eye(len(tempa)).to(tempa.device))
                        tempa = tempa @ tempa.T
                        loss_dict['disentangle_basis'] = torch.abs(tempa).sum()*0.001 

                    if isinstance(adapter_feat_list, list):
                        adapter_feat_list = torch.stack(adapter_feat_list, dim=0)
                    all_hoi_list = (target_classes_label.tolist() + unseen_list)
                    # adapted_weight = adapter_feat_list[:, all_hoi_list] @ torch.pinverse(self.basis_feat)
                    # adapted_weight = adapted_weight.mean(dim=0)
                    # adapted_weight = adapted_weight / adapted_weight.norm(dim=-1, keepdim=True)

                    origini_weight = self.hoicls_w / self.hoicls_w.norm(dim=-1, keepdim=True)
                        
                    if self.no_act_constraint is False:
                        # if self.zs_type in ['rare_first', 'non_rare_first'] and len(self.filtered_hoi_idx) > 0: 
                        #     loss_dict['loss_actw_similar'] = kl_loss(adapted_weight_actpart[all_hoi_list], act_weight[all_hoi_list]) * 50
                        # else:
                        all_logit = torch.cat(logits)
                        all_lb = torch.cat(labels_matched)
                        all_prior = torch.cat(prior, dim=1).prod(0)
                        ind_lg, ind_act = torch.nonzero(all_logit * all_lb).unbind(1)
                        pred_gt = all_logit[ind_lg, ind_act].sigmoid() * all_prior[ind_lg, ind_act]
                        conf_gt_act = torch.unique(ind_act[torch.nonzero(pred_gt > 0.5).squeeze(1)])
                        unconf_act = [i for i in target_classes_label.tolist() if HOI_TO_AO[i][0] not in conf_gt_act]
                        seenact_conf_w = [1/5 if i in conf_gt_act  else 3/5 for i in target_classes_label.tolist() ]   ### TODO to be finetuned
                        seenact_conf_w = torch.tensor(seenact_conf_w, device=adapter_feat_list.device)*50
                        if self.zs_type in ['default', 'rare_first']:
                            rare_list = [j for j in RARE_HOI_IDX if HOI_TO_AO[j][1] in [HOI_TO_AO[i.item()][1] for i in target_classes_label]]
                        else:
                            rare_list = []
             

                    if self.ao_sep_basis is True:
                        temp_actcls_w = self.actcls_w.clone().detach()
                        if self.basis_feat_constraint == 'none':
                            loss_dict['recon_loss2_act'] = self.recon_loss2(temp_actcls_w @ self.basis_feat[self.act_related_index], self.origin_text_embeddings.to(self.basis_feat.device)) * 0.1
                        else:
                            loss_dict['recon_loss2_act'] = self.recon_loss2(temp_actcls_w @ self.act_basis_feat, self.origin_text_embeddings.to(self.basis_feat.device)) * 0.1
                        
                        if self.basis_feat_constraint == 'l2':
                            loss_dict['loss2_basis_act'] = self.recon_loss2(self.act_basis_feat, self.basis_feat[self.act_related_index]) * 0.1
                        elif self.basis_feat_constraint == 'kl':
                            loss_dict['loss2_basis_act'] = kl_loss(self.act_basis_feat, self.basis_feat[self.act_related_index], T=self.kl_t) * 1E4

                        loss_dict['loss_actw_sparse'] = (torch.abs(self.actcls_w)).sum(-1).mean(0) * 0.1 
                        if self.disentangle_basis is True:
                            tempa = self.actcls_w / self.actcls_w.norm(dim=-1, keepdim=True)
                            tempa = tempa @ tempa.T 
                            tempa = tempa * (torch.ones_like(tempa).to(tempa.device) - torch.eye(len(tempa)).to(tempa.device))
                            tempa = tempa @ tempa.T
                            loss_dict['disentangle_basis_act'] = torch.abs(tempa).sum()*0.001 

                        
                        adapted_weight = (adapter_feat_list @ torch.pinverse(self.basis_feat)).mean(dim=0)
                        adapted_weight_actpart = adapted_weight[:, self.act_related_index] 
                        adapted_weight_actpart = adapted_weight_actpart / adapted_weight_actpart.norm(dim=-1, keepdim=True)
                        if self.dataset == 'hicodet':
                            act_weight = self.actcls_w[HOI_IDX_TO_ACT_IDX]
                        else:
                            act_weight = self.actcls_w[[HOI_TO_AO_COCO[i][0] for i in range(236)]]
                        act_weight = act_weight / act_weight.norm(dim=-1, keepdim=True)
                        if self.fix_act_w is True:
                            act_weight = act_weight.clone().detach()
                            
                        if self.no_act_constraint is False:
                            if len(unconf_act) > 0:
                                loss_dict['loss_actw_similar'] = (kl_loss(adapted_weight_actpart[target_classes_label.tolist()], act_weight[target_classes_label.tolist()], reduction='none', T=self.kl_t).sum(-1) * seenact_conf_w).sum() 
                            else:
                                loss_dict['loss_actw_similar'] = kl_loss(adapted_weight_actpart[target_classes_label.tolist()], act_weight[target_classes_label.tolist()], T=self.kl_t) * 50 /5
                            if len(unseen_list) > 0:
                                loss_dict['loss_actw_similar'] = loss_dict['loss_actw_similar'] + kl_loss(adapted_weight_actpart[unseen_list], act_weight[unseen_list], T=self.kl_t) * 50 * self.act_constraint_w_unseen
                            if len(rare_list) > 0:
                                loss_dict['loss_actw_similar'] = loss_dict['loss_actw_similar'] + kl_loss(adapted_weight_actpart[rare_list], act_weight[rare_list], T=self.kl_t) * 50

                        if self.seperate_ho != 0:
                            
                            ho_adapter_feat_l = torch.stack(ho_adapter_feat_list, dim=0).mean(dim=0)
                            ho_adapter_feat_l = ho_adapter_feat_l[:, :(ho_adapter_feat_l.shape[-1])//2]
                            if self.basis_feat_constraint == 'none':
                                if self.dataset == 'hicodet': 
                                    adapted_weight_human = ho_adapter_feat_l @ torch.pinverse(self.basis_feat[self.act_related_index])
                                else:
                                    adapted_weight_human = ho_adapter_feat_l @ torch.pinverse(self.basis_feat[self.act_related_index])   
                            else:
                                if self.dataset == 'hicodet':
                                    adapted_weight_human = ho_adapter_feat_l @ torch.pinverse(self.act_basis_feat)
                                else:
                                    adapted_weight_human = ho_adapter_feat_l @ torch.pinverse(self.act_basis_feat)
                            adapted_weight_human = adapted_weight_human / adapted_weight_human.norm(dim=-1, keepdim=True)

                            if self.no_act_constraint is False:
                                if len(unconf_act) > 0:
                                    loss_dict['loss_actw_similar2'] = (kl_loss(adapted_weight_human[target_classes_label.tolist()], act_weight[target_classes_label.tolist()], reduction='none', T=self.kl_t).sum(-1) * seenact_conf_w).sum()
                                    
                                else:
                                    loss_dict['loss_actw_similar2'] = kl_loss(adapted_weight_human[target_classes_label.tolist()], act_weight[target_classes_label.tolist()], T=self.kl_t) * 50 / 5 
                                if len(unseen_list) > 0:
                                    loss_dict['loss_actw_similar2'] = loss_dict['loss_actw_similar2'] + kl_loss(adapted_weight_human[unseen_list], act_weight[unseen_list], T=self.kl_t) * 50 * self.act_constraint_w_unseen
                                if len(rare_list) > 0:
                                    loss_dict['loss_actw_similar2'] = loss_dict['loss_actw_similar2'] + kl_loss(adapted_weight_human[rare_list], act_weight[rare_list], T=self.kl_t) * 50


                        
                return loss_dict
  
            if len(logits) == 0:
                print(targets)
                return None

            detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)

            return detections





def plot_tsne(all_features, HOII_l, object_labels, action_labels, title='t-SNE Visualization', savepth = 'temp.jpg'):
    
    feature_matrix = [ all_features[i]  for i in range(len(all_features)) if i in HOII_l]
    all_features = np.vstack(feature_matrix)

    # t-SNE 降维到2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    features_2d = tsne.fit_transform(all_features)

    # features_2d = [ features_2d[i] for i in range(len(features_2d)) if i in HOII_l]
    # features_2d = np.vstack(features_2d)
    # 设置颜色和形状的选择
    unique_objects = np.unique(object_labels)
    unique_actions = np.unique(action_labels)

    # 定义颜色映射和 marker 映射
    colors = ['r','g','b', 'y', 'cyan']
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', '<', '>', 'v']  # 常见 marker
    markers = markers[:len(unique_objects)]

    fig, ax = plt.subplots(figsize=(8,6))

    # 为每个样本绘制
    for i in range(len(features_2d)):
        obj_label_ind = unique_objects.tolist().index(object_labels[i])
        act_label_ind = unique_actions.tolist().index(action_labels[i])
        ax.scatter(
            features_2d[i, 0], 
            features_2d[i, 1], 
            color=colors[act_label_ind], 
            marker=markers[obj_label_ind], 
            edgecolor='k', 
            s=400
        )

    # 创建 object 图例
    action_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) 
                      for i in range(len(unique_actions))]
    unique_actions = [ACT_IDX_TO_ACT_NAME[i] for i in unique_actions]
    action_legend = ax.legend(action_handles, unique_actions, title='Action', loc='upper left', bbox_to_anchor=(1, 1))

    # 添加 object 图例
    ax.add_artist(action_legend)

    # 创建 action 图例，确保每个 action 对应一个 marker
    object_handles = [plt.Line2D([0], [0], marker=markers[j], color='w', markerfacecolor='gray', markersize=10, label=obj_to_name[unique_objects[j]]) 
                      for j in range(len(unique_objects))]
    ax.legend(handles=object_handles, title='Object', loc='upper left', bbox_to_anchor=(1, 0.6))

    # 设置图像标题和标签
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.savefig(savepth, dpi=300, bbox_inches='tight')
    plt.close()


def random_color():
    rdn = random.randint(1, 1000)
    b = int(rdn * 997) % 255
    g = int(rdn * 4447) % 255
    r = int(rdn * 6563) % 255
    return b, g, r

def cal_similarity(key_embeds,
                   ref_embeds,
                   method='cosine',
                   temperature=-1):
    assert method in ['dot_product', 'cosine', 'euclidean']

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros((key_embeds.size(0), ref_embeds.size(0)),
                           device=key_embeds.device)

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'euclidean':
        return euclidean_dist(key_embeds, ref_embeds)
    elif method == 'dot_product':
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
            return dists
        else:
            return torch.mm(key_embeds, ref_embeds.t())

def kl_loss(prediction, targets, reduction='sum', T = 0.1):
    
    return F.kl_div(F.log_softmax(prediction / T, dim=1),
             F.log_softmax(targets / T, dim=1),  # 1.2 0.1 0.2 0.3
             reduction=reduction, log_target=True) / prediction.numel()


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def get_multi_prompts(classnames):   ## https://github.com/openai/CLIP/blob/main/data/prompts.md, 
    templates = ['a photo of a person {}.',
                'a video of a person {}.',
                'a example of a person {}.',
                'a demonstration of a person {}.',
                'a photo of the person {}.',
                'a video of the person {}.',
                'a example of the person {}.', 
                'a demonstration of the person {}.',
                ]
    hico_texts = [' '.join(name.split(' ')[5:]) for name in classnames]
    all_texts_input = []
    for temp in templates:
        texts_input = torch.cat([clip.tokenize(temp.format(text)) for text in hico_texts ])
        all_texts_input.append(texts_input)
    all_texts_input = torch.stack(all_texts_input,dim=0)
    return all_texts_input



@torch.no_grad()
def get_origin_text_emb(args, clip_model, tgt_class_names, obj_class_names):
    use_templates = args.use_templates
    if use_templates == False:
        text_inputs = torch.cat([clip.tokenize(classname, context_length=77, truncate=True) for classname in tgt_class_names])
    elif use_templates:
        text_inputs = get_multi_prompts(tgt_class_names)
        bs_t, nums, c = text_inputs.shape
        text_inputs = text_inputs.view(-1, c)

    with torch.no_grad():
        origin_text_embedding = clip_model.encode_text(text_inputs)
    if use_templates:
        origin_text_embedding = origin_text_embedding.view(bs_t, nums, -1).mean(0)

    origin_text_embedding = origin_text_embedding / origin_text_embedding.norm(dim=-1, keepdim=True) # text embeddings of hoi 117*512 or 600*512

    if obj_class_names is not None:
        obj_text_inputs = torch.cat([clip.tokenize(obj_text) for obj_text in obj_class_names])
        with torch.no_grad():
            obj_text_embedding = clip_model.encode_text(obj_text_inputs)
            object_embedding = obj_text_embedding
            # obj_text_embedding = obj_text_embedding[hoi_obj_list,:]
        return origin_text_embedding, object_embedding
    else:
        return origin_text_embedding


def build_detector(args, class_corr, object_n_verb_to_interaction, clip_model_path, num_anno, verb2interaction=None):

    detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
            # pdb.set_trace()
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model']) 
        else:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])


    clip_state_dict = torch.load(clip_model_path, map_location="cpu").state_dict()
    clip_model = CLIP_models_adapter_prior2.build_model(state_dict=clip_state_dict, use_adapter=args.use_insadapter, adapter_pos=args.adapter_pos, adapter_num_layers=args.adapter_num_layers,pt_attn=args.pt_attn, pt_learn = args.pt_learn, pt_lyr = args.pt_lyr)

    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError
    model = CustomCLIP(args, classnames=classnames, clip_model=clip_model)

    obj_class_names = [obj[1] for obj in hico_text_label.hico_obj_text_label]

    if args.dataset == 'swig' and not args.LA and 'e' not in args.prior_type:
        origin_text_embeddings = None
        object_embedding = torch.rand(1000, 1)
    else:
        if args.act_txtdecrip is False:
            origin_text_embeddings, object_embedding = get_origin_text_emb(args, clip_model=clip_model, tgt_class_names=classnames, obj_class_names=obj_class_names)
        else:
            if args.dataset =='hicodet':
                file_path = ("hoi_txtdescrip/llama_action_descrip_hicodet.txt")
            elif args.dataset =='vcoco':
                file_path = ("hoi_txtdescrip/llama_action_descrip_vcoco.txt")
            
            act_txtdescrip = {}
            with open(file_path, 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines if line.rstrip()]
            for l_idx,line_i in enumerate(lines):
                if line_i[0].isdigit():
                    act = int(line_i)
                    cur_key = act
                    act_txtdescrip[cur_key] = ''
                elif line_i[0] == "*":
                    act_txtdescrip[cur_key] += line_i[2:] + '. '

            origin_text_embeddings, object_embedding = get_origin_text_emb(args, clip_model=clip_model, tgt_class_names=list(act_txtdescrip.values()), obj_class_names=obj_class_names)

        origin_text_embeddings = origin_text_embeddings.clone().detach()
        object_embedding = object_embedding.clone().detach()

        if args.llmtxt is True:
            if args.dataset == 'hicodet':
                f2 = open("hoi_txtdescrip/hico_HOI_descrip.txt","r")
                cls_descrip = []
                lines = f2.readlines()
                count = 0
                for line_i in lines:
                    if count %3 == 1:
                        cls_descrip.append(list(hico_text_label.hico_text_label.values())[int(count/3)] +\
                                            ":" + line_i.split(":")[1][:-1])
                    count += 1
            elif args.dataset == 'vcoco':
                file_path = ("hoi_txtdescrip/vcoco_hoi_text_description.txt")   
                cls_descrip = []
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    lines = [line.rstrip() for line in lines if line.rstrip()]

                hico_txt_description = {}
                cur_key = 0
                for l_idx,line_i in enumerate(lines):
                    if line_i[0] == '(' and line_i[1].isdigit():
                        
                        act, obj = line_i.split(",")
                        act = int(act[1:])
                        obj = int(obj[1:-1])
                        if args.dataset =='hicodet':
                            hoii = MAP_AO_TO_HOI[act, obj]
                        else:
                            hoii = MAP_AO_TO_HOI_COCO[(act, obj)]
    
                        cur_key = hoii
                        hico_txt_description[cur_key] = ''
                    else:
                        hico_txt_description[cur_key] += line_i + ' '

                for  hoii in (hico_txt_description):
                    tnt_dep = hico_txt_description[hoii][:300]
                    quit_len = len(tnt_dep.split(".")[-1])
                    if quit_len > 0:
                        tnt_dep = tnt_dep[:-quit_len]
                    if args.dataset =='hicodet':
                        cls_descrip.append(list(hico_text_label.hico_text_label.values())[int(hoii)] +\
                                            ":" +tnt_dep)
    
                    else:
                        cls_descrip.append((vcoco_hoi_text_label)[HOI_TO_AO_COCO[int(hoii)]] +\
                                            ":" +tnt_dep)
            hoicls_txt, _ = get_origin_text_emb(args, clip_model=clip_model, tgt_class_names=cls_descrip, obj_class_names=obj_class_names)
        else:
            hoicls_txt, _ = get_origin_text_emb(args, clip_model=clip_model, tgt_class_names=list(hico_text_label.hico_text_label.values()), obj_class_names=obj_class_names)
    
        if args.ho_pair_pt is True:
            with open("hoi_txtdescrip/hico_ho_corepoint.jsonl", "r") as f:
                round1_ans = [(json.loads(line))['response']['body']['choices'][0]['message']['content'].split("\n") for line in f]

            HO_pair_text = {}
            for l_idx,ans_i in enumerate(round1_ans):
                HO_pair_text[l_idx] = {}
                flag = -1
                for line_i in ans_i:
                    line_i_temp = line_i.strip()
                    if len(line_i_temp) == 0:
                        continue
                    if "human body description" in line_i_temp.lower():
                        HO_pair_text[l_idx]["human"] = []
                        flag = 0
                    elif "object description" in line_i_temp.lower():
                        HO_pair_text[l_idx]["object"] = []
                        flag = 1
      
                    else:
                        if flag == 0:
                            HO_pair_text[l_idx]["human"].append(line_i_temp)
                        elif flag == 1:
                            try:
                                HO_pair_text[l_idx]["object"].append(line_i_temp)
                            except:
                                pdb.set_trace()

            HO_pair_text_embeddings = []
            for key in HO_pair_text:
                HO_pair_text_inputs = torch.cat([clip.tokenize(text_i, context_length=77, truncate=True) for idx, text_i in enumerate(list(HO_pair_text[key].values()))]) 

                with torch.no_grad():
                    HO_pair_text_embedding = clip_model.encode_text(HO_pair_text_inputs)
                    HO_pair_text_embedding = HO_pair_text_embedding / HO_pair_text_embedding.norm(dim=-1, keepdim=True)
                HO_pair_text_embeddings.append(HO_pair_text_embedding)

        else:
            HO_pair_text_embeddings = None

        args.feat_dim = hoicls_txt.shape[-1]        
        hoicls_txt = hoicls_txt.clone().detach()

    detector = UPT(args,
        detr, postprocessors['bbox'], model, origin_text_embeddings, object_embedding,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
        num_anno = num_anno,
        hoicls_txt = hoicls_txt,
        HO_pair_text_embeddings = HO_pair_text_embeddings
    )
    return detector