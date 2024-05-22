import types
import torch
import transformers
import torch.nn.functional as F
from torch import TensorType, nn
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers.modeling_outputs import BaseModelOutput
from torchtyping import TensorType
from typing import Optional
import math
import modeling_t5
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union


class LoRAT5(modeling_t5.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_lora()
        self.gen = False
        # self.encoder.block[0].layer[0].SelfAttention.gen = False

    def emb(self, l):
        """
        PCA Embedding of linguistic attestation vector
        """
        feature = l
        if not isinstance(l, torch.Tensor):
            feature = torch.Tensor(l)
        return feature

    def set_features(self, features):
        self.hypernet.down_hypernet.set_features(self.emb(features))
        self.hypernet.up_hypernet.set_features(self.emb(features))

        self.decoder_hypernet.down_hypernet.set_features(self.emb(features))
        self.decoder_hypernet.up_hypernet.set_features(self.emb(features))
        self.cross_hypernet.down_hypernet.set_features(self.emb(features))
        self.cross_hypernet.up_hypernet.set_features(self.emb(features))

        if 'ko' in self.config.name:
            self.hypernet_ko.down_hypernet.set_features(self.emb(features))
            self.hypernet_ko.up_hypernet.set_features(self.emb(features))
            self.decoder_hypernet_ko.down_hypernet.set_features(
                self.emb(features))
            self.decoder_hypernet_ko.up_hypernet.set_features(
                self.emb(features))
            self.cross_hypernet_ko.down_hypernet.set_features(
                self.emb(features))
            self.cross_hypernet_ko.up_hypernet.set_features(self.emb(features))
        if 'hyfn' in self.config.name:
            self.ffn_en_hypernet_wi.down_hypernet.set_features(
                self.emb(features))
            self.ffn_en_hypernet_wi.up_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wi.down_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wi.up_hypernet.set_features(
                self.emb(features))
            self.ffn_en_hypernet_wo.down_hypernet.set_features(
                self.emb(features))
            self.ffn_en_hypernet_wo.up_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wo.down_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wo.up_hypernet.set_features(
                self.emb(features))

    def set_prefixs(self, features):
        self.hypernet.down_hypernet.set_prefixs(self.emb(features))
        self.hypernet.up_hypernet.set_prefixs(self.emb(features))

        self.decoder_hypernet.down_hypernet.set_prefixs(self.emb(features))
        self.decoder_hypernet.up_hypernet.set_prefixs(self.emb(features))
        self.cross_hypernet.down_hypernet.set_prefixs(self.emb(features))
        self.cross_hypernet.up_hypernet.set_prefixs(self.emb(features))

        if 'ko' in self.config.name:
            self.hypernet_ko.down_hypernet.set_prefixs(self.emb(features))
            self.hypernet_ko.up_hypernet.set_prefixs(self.emb(features))
            self.decoder_hypernet_ko.down_hypernet.set_prefixs(
                self.emb(features))
            self.decoder_hypernet_ko.up_hypernet.set_prefixs(
                self.emb(features))
            self.cross_hypernet_ko.down_hypernet.set_prefixs(
                self.emb(features))
            self.cross_hypernet_ko.up_hypernet.set_prefixs(self.emb(features))
        if 'hyfn' in self.config.name:
            self.ffn_en_hypernet_wi.down_hypernet.set_prefixs(
                self.emb(features))
            self.ffn_en_hypernet_wi.up_hypernet.set_prefixs(self.emb(features))
            self.ffn_de_hypernet_wi.down_hypernet.set_prefixs(
                self.emb(features))
            self.ffn_de_hypernet_wi.up_hypernet.set_prefixs(self.emb(features))
            self.ffn_en_hypernet_wo.down_hypernet.set_prefixs(
                self.emb(features))
            self.ffn_en_hypernet_wo.up_hypernet.set_prefixs(self.emb(features))
            self.ffn_de_hypernet_wo.down_hypernet.set_prefixs(
                self.emb(features))
            self.ffn_de_hypernet_wo.up_hypernet.set_prefixs(self.emb(features))
            
    def set_inputs(self, features):
        self.hypernet.down_hypernet.set_inputs(features)
        self.hypernet.up_hypernet.set_inputs(features)

        self.decoder_hypernet.down_hypernet.set_inputs(features)
        self.decoder_hypernet.up_hypernet.set_inputs(features)
        self.cross_hypernet.down_hypernet.set_inputs(features)
        self.cross_hypernet.up_hypernet.set_inputs(features)

        if 'ko' in self.config.name:
            self.hypernet_ko.down_hypernet.set_inputs(features)
            self.hypernet_ko.up_hypernet.set_inputs(features)
            self.decoder_hypernet_ko.down_hypernet.set_inputs(features)
            self.decoder_hypernet_ko.up_hypernet.set_inputs(features)
            self.cross_hypernet_ko.down_hypernet.set_inputs(features)
            self.cross_hypernet_ko.up_hypernet.set_inputs(features)
        if 'hyfn' in self.config.name:
            self.ffn_en_hypernet_wi.down_hypernet.set_inputs(features)
            self.ffn_en_hypernet_wi.up_hypernet.set_inputs(features)
            self.ffn_de_hypernet_wi.down_hypernet.set_inputs(features)
            self.ffn_de_hypernet_wi.up_hypernet.set_inputs(features)
            self.ffn_en_hypernet_wo.down_hypernet.set_inputs(features)
            self.ffn_en_hypernet_wo.up_hypernet.set_inputs(features)
            self.ffn_de_hypernet_wo.down_hypernet.set_inputs(features)
            self.ffn_de_hypernet_wo.up_hypernet.set_inputs(features)

    def forward(self, input_ids=None, attention_mask=None, labels=None, features=None,
                instruction_input=None, instruction_attention_mask=None, original_mask=None,
                original_embedding=None, include_original=False, **kwargs):
        """
        forward model needs to include features parameter for Trainer to not discard this feature
        """
        inputs = {"labels": labels, "input_ids": input_ids,
                  "attention_mask": attention_mask, **kwargs}
        if include_original:
            inputs["original_embedding"] = original_embedding
            inputs["original_mask"] = original_mask

        if not self.config.whitening and not self.gen:
            if self.config.hyperencoder:
                hidden_states = self.hyperencoder(
                    **features, return_dict=True, output_hidden_states=True).hidden_states
                instruction_input = hidden_states[-1]
                instruction_attention_mask = features["attention_mask"]
                features = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            features = self.adap_pooler(features)

        if self.gen:
            features = self.features
            instruction_input = self.instruction_input_gen
            instruction_attention_mask = self.instruction_attention_mask_gen

        self.instruction_input = instruction_input
        self.instruction_attention_mask = instruction_attention_mask

        self.set_features(features)
        if "prefix" in self.config.name:
            if "gpt" in self.config.name:
                self.set_inputs(instruction_input)
            else:
                self.set_prefixs(instruction_input)

        return super().forward(**inputs)

    def generate(self, input_ids, attention_mask, features=None, instruction_input=None, instruction_attention_mask=None, **kwargs):
        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask,
                  **kwargs}

        if not self.config.whitening:
            if self.config.hyperencoder:
                hidden_states = self.hyperencoder(
                    **features, return_dict=True, output_hidden_states=True).hidden_states
                instruction_input = hidden_states[-1]
                instruction_attention_mask = features["attention_mask"]
                features = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            features = self.adap_pooler(features)

        self.features = features
        self.instruction_input_gen = instruction_input
        self.instruction_attention_mask_gen = instruction_attention_mask

        self.set_features(features)
        if "prefix" in self.config.name:
            self.set_prefixs(instruction_input)

        return super().generate(**inputs)

    def load_t5(self, state_dict):
        self.unwrap_lora()
        self.load_state_dict(state_dict)
        self.wrap_lora()
        self.freeze_params()

    def unwrap_lora(self):
        for i, l in enumerate(self.encoder.block):
            l = l.module if hasattr(l, "module") else l
            l.layer[0].SelfAttention.q = l.layer[0].SelfAttention.q.linear
            l.layer[0].SelfAttention.v = l.layer[0].SelfAttention.v.linear

        for i, l in enumerate(self.decoder.block):
            l = l.module if hasattr(l, "module") else l
            l.layer[0].SelfAttention.q = l.layer[0].SelfAttention.q.linear
            l.layer[0].SelfAttention.v = l.layer[0].SelfAttention.v.linear
            l.layer[1].EncDecAttention.q = l.layer[1].EncDecAttention.q.linear
            l.layer[1].EncDecAttention.v = l.layer[1].EncDecAttention.v.linear

        if 'ko' in self.config.name:
            for i, l in enumerate(self.encoder.block):
                l = l.module if hasattr(l, "module") else l
                l.layer[0].SelfAttention.k = l.layer[0].SelfAttention.k.linear
                l.layer[0].SelfAttention.o = l.layer[0].SelfAttention.o.linear

            for i, l in enumerate(self.decoder.block):
                l = l.module if hasattr(l, "module") else l
                l.layer[0].SelfAttention.k = l.layer[0].SelfAttention.k.linear
                l.layer[0].SelfAttention.o = l.layer[0].SelfAttention.o.linear
                l.layer[1].EncDecAttention.k = l.layer[1].EncDecAttention.k.linear
                l.layer[1].EncDecAttention.o = l.layer[1].EncDecAttention.o.linear

        if 'hyfn' in self.config.name:
            for i, l in enumerate(self.encoder.block):
                l = l.module if hasattr(l, "module") else l
                l.layer[1].DenseReluDense.wi_0 = l.layer[1].DenseReluDense.wi_0.linear
                l.layer[1].DenseReluDense.wi_1 = l.layer[1].DenseReluDense.wi_1.linear
                l.layer[1].DenseReluDense.wo = l.layer[1].DenseReluDense.wo.linear
            for i, l in enumerate(self.decoder.block):
                l = l.module if hasattr(l, "module") else l
                l.layer[2].DenseReluDense.wi_0 = l.layer[2].DenseReluDense.wi_0.linear
                l.layer[2].DenseReluDense.wi_1 = l.layer[2].DenseReluDense.wi_1.linear
                l.layer[2].DenseReluDense.wo = l.layer[2].DenseReluDense.wo.linear
        for key in dir(self):
            if 'hypernet' in key or 'pooler' in key or 'hyperencoder' in key or 'prefix_gen' in key:
                delattr(self, key)

    def wrap_lora(self):
        """
        Wrap T5 to obtain a hypernetwork lora model.
        """
        if self.config.hyperencoder:
            self.hyperencoder = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                'google/t5-base-lm-adapt',
                torch_dtype=torch.bfloat16
                # cache_dir=model_args.cache_dir,
            ).encoder
            self.config.pooler_d_model = 768
        if "gpt" in self.config.name:
            self.prefix_gen = MyGPT2.from_pretrained(
                'openai-community/gpt2',
                torch_dtype=torch.bfloat16
                # cache_dir=model_args.cache_dir,
            )
            self.prefix_gen._init_prompt(self.config.prefix_length, self.config.d_model)
        else:
            self.prefix_gen = None
        self.down_hypernet = None
        self.up_hypernet = None
        self.embedding_dim = self.config.embedding_dim
        self.encoding_dim = self.config.encoding_dim
        down_dim = self.config.d_kv * self.config.num_heads
        input_dim = self.config.d_model
        self.adap_pooler = MyPooler(
            self.config.pooler_d_model, self.encoding_dim)

        self.hypernet = HyperNet(
            self.encoding_dim, input_dim, self.embedding_dim, down_dim)

        self.decoder_hypernet = HyperNet(
            self.encoding_dim, input_dim, self.embedding_dim, down_dim)

        self.cross_hypernet = HyperNet(
            self.encoding_dim, input_dim, self.embedding_dim, down_dim)

        if 'ko' in self.config.name:
            self.hypernet_ko = HyperNet(
                self.encoding_dim, input_dim, self.embedding_dim, down_dim)
            self.decoder_hypernet_ko = HyperNet(
                self.encoding_dim, input_dim, self.embedding_dim, down_dim)
            self.cross_hypernet_ko = HyperNet(
                self.encoding_dim, input_dim, self.embedding_dim, down_dim)

        if 'hyfn' in self.config.name:
            self.ffn_en_hypernet_wi = HyperNet(
                self.encoding_dim, input_dim, self.embedding_dim * 8, down_dim * 4)
            self.ffn_en_hypernet_wo = HyperNet(
                self.encoding_dim, input_dim * 4, self.embedding_dim * 8, down_dim)
            self.ffn_de_hypernet_wi = HyperNet(
                self.encoding_dim, input_dim, self.embedding_dim * 8, down_dim * 4)
            self.ffn_de_hypernet_wo = HyperNet(
                self.encoding_dim, input_dim * 4, self.embedding_dim * 8, down_dim)

        self.init_hyper()

    def freeze_params(self):
        if 'hyperlora' in self.config.name:
            for layer in self.modules():
                for _, param in layer.named_parameters():
                    param.requires_grad = False
        else:
            # All modules in the
            # modules_to_freeze = [self.model.encoder.encoder.block[i].layer[0] for i in range(len(self.model.encoder.encoder.block))]
            modules_to_freeze = [l.module.layer[0] if hasattr(
                l, "module") else l.layer[0] for l in self.model.encoder.encoder.block]
            # modules_to_freeze.extend([l.module.layer[1] if hasattr(l, "module") else l.layer[1] for l in self.model.encoder.encoder.block])
            # And the decoder modules, which has both a SelfAttention (layer[0])
            modules_to_freeze.extend([self.model.decoder.block[i].layer[0] for i in range(
                len(self.model.decoder.block))])
            # and CrossAttention (layer[1]) block
            modules_to_freeze.extend([self.model.decoder.block[i].layer[1] for i in range(
                len(self.model.decoder.block))])
            # modules_to_freeze.extend([self.model.decoder.block[i].layer[2] for i in range(len(self.model.decoder.block))])
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False  # Actual freezing operation
        for layer in self.modules():
            for x, param in layer.named_parameters():
                if "norm" in x or "emb" in x or "hypernet" in x or "pooler" in x or "hyperencoder" in x or "prefix" in x:
                    param.requires_grad = True
        if 'ffn' in self.config.name:
            for layer in self.modules():
                for x, param in layer.named_parameters():
                    if "wi" in x or "wo" in x:
                        param.requires_grad = True

    def init_hyper(self):
        for i, l in enumerate(self.encoder.block):
            l = l.module if hasattr(l, "module") else l
            l.layer[0].SelfAttention.q = HyperLora(
                l.layer[0].SelfAttention.q, self.hypernet.down_hypernet, self.hypernet.up_hypernet, 2*i)
            if 'prefix' in self.config.name:
                l.layer[0].SelfAttention.v = HyperLora(l.layer[0].SelfAttention.v, self.hypernet.down_hypernet, self.hypernet.up_hypernet, 2*i+1,
                                                       self.config.pooler_d_model, self.config.prefix_length, self.config.max_source_length, self.config.d_model, self.prefix_gen)
            else:
                l.layer[0].SelfAttention.v = HyperLora(
                    l.layer[0].SelfAttention.v, self.hypernet.down_hypernet, self.hypernet.up_hypernet, 2*i+1)
            if 'ko' in self.config.name:
                if 'prefix' in self.config.name:
                    l.layer[0].SelfAttention.k = HyperLora(l.layer[0].SelfAttention.k, self.hypernet_ko.down_hypernet, self.hypernet_ko.up_hypernet, 2*i,
                                                           self.config.pooler_d_model, self.config.prefix_length, self.config.max_source_length, self.config.d_model, self.prefix_gen)
                else:
                    l.layer[0].SelfAttention.k = HyperLora(
                        l.layer[0].SelfAttention.k, self.hypernet_ko.down_hypernet, self.hypernet_ko.up_hypernet, 2*i)
                l.layer[0].SelfAttention.o = HyperLora(
                    l.layer[0].SelfAttention.o, self.hypernet_ko.down_hypernet, self.hypernet_ko.up_hypernet, 2*i+1)

        for i, l in enumerate(self.decoder.block):
            l = l.module if hasattr(l, "module") else l
            l.layer[0].SelfAttention.q = HyperLora(
                l.layer[0].SelfAttention.q, self.decoder_hypernet.down_hypernet, self.decoder_hypernet.up_hypernet, 2*i)
            l.layer[1].EncDecAttention.q = HyperLora(
                l.layer[1].EncDecAttention.q, self.cross_hypernet.down_hypernet, self.cross_hypernet.up_hypernet, 2*i)
            if 'prefix' in self.config.name:
                l.layer[0].SelfAttention.v = HyperLora(l.layer[0].SelfAttention.v, self.decoder_hypernet.down_hypernet, self.decoder_hypernet.up_hypernet, 2*i+1,
                                                       self.config.pooler_d_model, self.config.prefix_length, self.config.max_source_length, self.config.d_model, self.prefix_gen)
                l.layer[1].EncDecAttention.v = HyperLora(l.layer[1].EncDecAttention.v, self.cross_hypernet.down_hypernet, self.cross_hypernet.up_hypernet, 2*i+1,
                                                         self.config.pooler_d_model, self.config.prefix_length, self.config.max_source_length, self.config.d_model, self.prefix_gen)
            else:
                l.layer[0].SelfAttention.v = HyperLora(
                    l.layer[0].SelfAttention.v, self.decoder_hypernet.down_hypernet, self.decoder_hypernet.up_hypernet, 2*i+1)
                l.layer[1].EncDecAttention.v = HyperLora(
                    l.layer[1].EncDecAttention.v, self.cross_hypernet.down_hypernet, self.cross_hypernet.up_hypernet, 2*i+1)
            if 'ko' in self.config.name:
                if 'prefix' in self.config.name:
                    l.layer[0].SelfAttention.k = HyperLora(l.layer[0].SelfAttention.k, self.decoder_hypernet_ko.down_hypernet, self.decoder_hypernet_ko.up_hypernet, 2*i,
                                                           self.config.pooler_d_model, self.config.prefix_length, self.config.max_source_length, self.config.d_model, self.prefix_gen)
                    l.layer[1].EncDecAttention.k = HyperLora(l.layer[1].EncDecAttention.k, self.cross_hypernet_ko.down_hypernet, self.cross_hypernet_ko.up_hypernet, 2*i,
                                                             self.config.pooler_d_model, self.config.prefix_length, self.config.max_source_length, self.config.d_model, self.prefix_gen)
                else:
                    l.layer[0].SelfAttention.k = HyperLora(
                        l.layer[0].SelfAttention.k, self.decoder_hypernet_ko.down_hypernet, self.decoder_hypernet_ko.up_hypernet, 2*i)
                    l.layer[1].EncDecAttention.k = HyperLora(
                        l.layer[1].EncDecAttention.k, self.cross_hypernet_ko.down_hypernet, self.cross_hypernet_ko.up_hypernet, 2*i)
                l.layer[0].SelfAttention.o = HyperLora(
                    l.layer[0].SelfAttention.o, self.decoder_hypernet_ko.down_hypernet, self.decoder_hypernet_ko.up_hypernet, 2*i+1)
                l.layer[1].EncDecAttention.o = HyperLora(
                    l.layer[1].EncDecAttention.o, self.cross_hypernet_ko.down_hypernet, self.cross_hypernet_ko.up_hypernet, 2*i+1)

        if 'hyfn' in self.config.name:
            for i, l in enumerate(self.encoder.block):
                l = l.module if hasattr(l, "module") else l
                l.layer[1].DenseReluDense.wi_0 = HyperLora(
                    l.layer[1].DenseReluDense.wi_0, self.ffn_en_hypernet_wi.down_hypernet, self.ffn_en_hypernet_wi.up_hypernet, 2*i)
                l.layer[1].DenseReluDense.wi_1 = HyperLora(
                    l.layer[1].DenseReluDense.wi_1, self.ffn_en_hypernet_wi.down_hypernet, self.ffn_en_hypernet_wi.up_hypernet, 2*i+1)
                l.layer[1].DenseReluDense.wo = HyperLora(
                    l.layer[1].DenseReluDense.wo, self.ffn_en_hypernet_wo.down_hypernet, self.ffn_en_hypernet_wo.up_hypernet, i)
            for i, l in enumerate(self.decoder.block):
                l = l.module if hasattr(l, "module") else l
                l.layer[2].DenseReluDense.wi_0 = HyperLora(
                    l.layer[2].DenseReluDense.wi_0, self.ffn_de_hypernet_wi.down_hypernet, self.ffn_de_hypernet_wi.up_hypernet, 2*i)
                l.layer[2].DenseReluDense.wi_1 = HyperLora(
                    l.layer[2].DenseReluDense.wi_1, self.ffn_de_hypernet_wi.down_hypernet, self.ffn_de_hypernet_wi.up_hypernet, 2*i+1)
                l.layer[2].DenseReluDense.wo = HyperLora(
                    l.layer[2].DenseReluDense.wo, self.ffn_de_hypernet_wo.down_hypernet, self.ffn_de_hypernet_wo.up_hypernet, i)


class MyGPT2(transformers.GPT2Model):

    def _init_prompt(self, d_len, hidden_size):
        hypernet = nn.Embedding(d_len, hidden_size)
        hypernet.weight.data.normal_(mean=0.0)
        self.init_prompt_weight = hypernet(torch.arange(0, d_len))
    
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            demo_token_type_ids=None,
            **kwargs
    ):
        if hasattr(self, "init_prompt_weight"):
            distill_embeds = self.init_prompt_weight.unsqueeze(0).expand(input_ids.shape[0], -1, -1).to(input_ids.device)
            inputs_embeds = torch.cat([self.get_input_embeddings()(input_ids.int()), distill_embeds], dim=1).to(torch.bfloat16)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(distill_embeds.shape[0], distill_embeds.shape[1])], dim=-1).to(torch.bfloat16)
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        last_hidden_states = super().forward(
            None,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            output_attentions,
            output_hidden_states=True).last_hidden_state

        if hasattr(self, "init_prompt_weight"):
            distill_embeds_shape = self.init_prompt_weight.shape
            last_hidden_states = last_hidden_states[:, -distill_embeds_shape[0]:, :]
            attention_mask = attention_mask[:, -distill_embeds_shape[0]:]
        return last_hidden_states


class HyperParamNet(nn.Module):
    """
    Helper class to wrap together hypernetwork weights "linear1" and "linear2" into MLP

    Arguments:
    - linear1 : Downsampling weight (encoding dim x bottleneck)
    - linear2 : Upsampling weight (bottleneck x dim)
    - dim : output dim
    - bottleneck : bottleneck dim

    Output:
    Adapter weight generated by hypernetworks with dialect feature input
    """

    def __init__(self, linear1, linear2, dim, bottleneck, scale=False):
        super().__init__()
        self.linear1 = linear1
        self.linear2 = linear2
        self.dim = dim  # Output dimension
        self.bottleneck = bottleneck  # MLP bottleneck
        self.features = None
        if scale:
            self.scale = math.sqrt(dim)
        else:
            self.scale = 1

    def set_features(self, features):
        self.features = features
        return

    def set_prefixs(self, prefixs):
        self.prefixs = prefixs
        return
    
    def set_inputs(self, inputs):
        self.inputs = inputs
        return

    def forward(self, features):
        output = self.linear2(F.relu(self.linear1(features))).reshape(-1, self.dim, self.bottleneck)
        return output/self.scale


class HyperLora(nn.Module):
    """
    Simple MLP Hypernet
    """

    def __init__(self, linear: nn.Module, hypernet1=None, hypernet2=None, idx=0, feature_size=0, prefix_length=0, src_length=0, target_size=0, prefix_gen=None):
        super().__init__()

        self.linear = linear
        self.hypernet1 = hypernet1
        self.hypernet2 = hypernet2
        self.dropout = nn.Dropout(p=0.1)
        # Layer idx
        self.idx = idx
        # prefix
        self.prefix = False
        self.gpt = False
        if prefix_length > 0:
            self.prefix = True
            self.gpt = prefix_gen is not None
            # self.prefix_gen = hypernetwork(
            #     target_size * prefix_length, "prefix", feature_size) if prefix_gen is None else prefix_gen
            self.prefix_gen = MyPrefix(
                feature_size, prefix_length, src_length, target_size)

    def forward(self, x):
        # Conditioning variable (either indicator or example)
        val = self.hypernet1.features
        # Layer idx is added to conditioning variable
        if self.idx is not None:
            val = nn.functional.pad(val, (0, 1), value=self.idx)

        val = val.to(torch.bfloat16)
        # Compute hypernet weights
        self.lora_A = self.hypernet1(val)
        self.lora_B = self.hypernet2(val)
        
        # weight1 = weight1.repeat(
        #     1, x.shape[0] // weight1.shape[0], 1).view(-1, weight1.shape[1], weight1.shape[2])
        # weight2 = weight2.repeat(
        #     1, x.shape[0] // weight2.shape[0], 1).view(-1, weight2.shape[1], weight2.shape[2])
        # Apply lora
        out = self.dropout(self.linear(x))
        out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B) + out
        if self.prefix:
            # **self.hypernet1.inputs if self.gpt else self.hypernet1.prefixs
            prefix = self.prefix_gen(self.hypernet1.prefixs)
            out = torch.cat((prefix, out), dim=1)
        return out


class HyperNet(nn.Module):
    def __init__(self, encoding_dim, input_dim, embedding_dim, output_dim):
        super(HyperNet, self).__init__()
        self.hidden_dim = 8
        self.pre_down_linear = nn.Linear(encoding_dim+1, self.hidden_dim)
        self.pre_down_linear.weight, self.pre_down_linear.bias = self.init_layer(
            self.pre_down_linear)
        self.down_linear = nn.Linear(self.hidden_dim, input_dim*embedding_dim)
        self.down_linear.weight, self.down_linear.bias = self.init_layer(
            self.down_linear)
        self.pre_up_linear = nn.Linear(encoding_dim+1, self.hidden_dim)
        self.pre_up_linear.weight, self.pre_up_linear.bias = self.init_layer(
            self.pre_up_linear)
        self.up_linear = nn.Linear(self.hidden_dim, embedding_dim*output_dim)
        self.up_linear.weight, self.up_linear.bias = self.init_layer(
            self.up_linear)

        self.down_hypernet = HyperParamNet(
            self.pre_down_linear, self.down_linear, input_dim, embedding_dim)
        self.up_hypernet = HyperParamNet(
            self.pre_up_linear, self.up_linear, embedding_dim, output_dim, scale=True)

    def init_layer(self, layer, bias=True):
        weight = nn.Parameter(torch.normal(0, 1e-7, layer.weight.shape))
        if bias:
            bias = nn.init.zeros_(layer.bias)
        else:
            bias = None
        return weight, bias


class MyPrefix(nn.Module):
    def __init__(self, feature_size, prefix_length, src_length, target_size):
        super().__init__()
        middle_size = 128
        self.target_size = target_size
        # self.maxpool = nn.MaxPool2d((src_length - prefix_length + 1, 1), stride=1)
        self.dense = nn.Linear(src_length, prefix_length)
        # if feature_size != target_size:
        self.down = nn.Linear(feature_size, middle_size)
        self.up = nn.Linear(middle_size, target_size)
        # self.down = nn.Linear(feature_size, middle_size)
        # self.dense = nn.Linear(middle_size * src_length, prefix_length * middle_size)
        # self.up = nn.Linear(middle_size, target_size)
        self.layernorm = nn.LayerNorm(target_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, mask=None) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.bfloat16)  # b, l, h
        # flat_hidden_states = self.maxpool(down_states)
        down_states = self.down(hidden_states)
        flat_hidden_states = down_states.reshape(
            hidden_states.shape[0], down_states.shape[-1], -1)
        pooled_output = self.dense(flat_hidden_states)
        pooled_output = pooled_output.reshape(
            hidden_states.shape[0], -1, down_states.shape[-1])
        pooled_output = self.layernorm(self.up(pooled_output))
        pooled_output = self.dropout(F.relu(pooled_output))
        # if hidden_states.shape[-1] != self.target_size:
        #     down_states = self.activation(self.down(pooled_output))
        #     pooled_output = self.dropout(self.activation(self.up(down_states)))
        return pooled_output


class MyPooler(nn.Module):
    def __init__(self, s_hidden_size, t_hidden_size):
        super().__init__()
        self.dense = nn.Linear(s_hidden_size, t_hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.bfloat16)
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        pooled_output = self.dense(flat_hidden_states)
        pooled_output = self.activation(pooled_output)
        nested_output = pooled_output.reshape(*hidden_states.shape[:-1], -1)
        return nested_output


class AdapterWrapper(nn.Module):
    """
    General Wrapper Class for Hypernet Config

    Each child class needs to implement the init hypernet method that injects hypernet weights
    """

    def __init__(self, model, embedding_dim, weights, args):
        super(AdapterWrapper, self).__init__()
        self.model = model
        self.down_hypernet = None
        self.up_hypernet = None
        self.embedding_dim = embedding_dim
        self.encoding_dim = args.encoding_dim
        self.args = args
        self.config = model.config
        down_dim = model.config.d_kv * model.config.num_heads
        input_dim = model.config.d_model
        self.adap_pooler = MyPooler(args.d_model, self.encoding_dim)
        self.model.custom_model = args.custom_model

        self.hypernet = HyperNet(
            self.encoding_dim, input_dim, self.embedding_dim, down_dim)

        if 'hyperlora' in self.args.name:
            self.decoder_hypernet = HyperNet(
                self.encoding_dim, input_dim, self.embedding_dim, down_dim)

            self.cross_hypernet = HyperNet(
                self.encoding_dim, input_dim, self.embedding_dim, down_dim)

        if 'hyfn' in self.args.name:
            self.ffn_en_hypernet_wi = HyperNet(
                self.encoding_dim, input_dim, self.embedding_dim * 8, down_dim * 4)
            self.ffn_en_hypernet_wo = HyperNet(
                self.encoding_dim, input_dim * 4, self.embedding_dim * 8, down_dim)
            self.ffn_de_hypernet_wi = HyperNet(
                self.encoding_dim, input_dim, self.embedding_dim * 8, down_dim * 4)
            self.ffn_de_hypernet_wo = HyperNet(
                self.encoding_dim, input_dim * 4, self.embedding_dim * 8, down_dim)

        if weights is not None:
            self.hypernet.load_state_dict(torch.load(
                weights, map_location=torch.device('cpu'), strict=False))
            print("WEIGHTS LOADED")

        # self.earth_mover_loss = SamplesLoss(loss="sinkhorn", p=2)

        self.init_hypernet()

    def init_layer(self, layer):
        weight = nn.Parameter(torch.normal(0, 1e-7, layer.weight.shape))
        bias = nn.init.zeros_(layer.bias)
        return weight, bias

    def init_hypernet(self):
        pass

    def freeze_params(self):
        if 'hyperlora' in self.args.name:
            for layer in self.model.modules():
                for _, param in layer.named_parameters():
                    param.requires_grad = False
        else:
            # All modules in the
            # modules_to_freeze = [self.model.encoder.encoder.block[i].layer[0] for i in range(len(self.model.encoder.encoder.block))]
            modules_to_freeze = [l.module.layer[0] if hasattr(
                l, "module") else l.layer[0] for l in self.model.encoder.encoder.block]
            # modules_to_freeze.extend([l.module.layer[1] if hasattr(l, "module") else l.layer[1] for l in self.model.encoder.encoder.block])
            # And the decoder modules, which has both a SelfAttention (layer[0])
            modules_to_freeze.extend([self.model.decoder.block[i].layer[0] for i in range(
                len(self.model.decoder.block))])
            # and CrossAttention (layer[1]) block
            modules_to_freeze.extend([self.model.decoder.block[i].layer[1] for i in range(
                len(self.model.decoder.block))])
            # modules_to_freeze.extend([self.model.decoder.block[i].layer[2] for i in range(len(self.model.decoder.block))])
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False  # Actual freezing operation
        for layer in self.model.modules():
            for x, param in layer.named_parameters():
                if "norm" in x or "emb" in x or "hypernet" in x or "pooler" in x:
                    param.requires_grad = True
        if 'ffn' in self.args.name:
            for layer in self.model.modules():
                for x, param in layer.named_parameters():
                    if "wi" in x or "wo" in x:
                        param.requires_grad = True
            # self.hypernet.pre_down_linear.weight.requires_grad = True
            # self.hypernet.pre_down_linear.bias.requires_grad = True
            # self.hypernet.pre_up_linear.weight.requires_grad = True
            # self.hypernet.pre_up_linear.bias.requires_grad = True
            # self.hypernet.down_linear.weight.requires_grad = True
            # self.hypernet.down_linear.bias.requires_grad = True
            # self.hypernet.up_linear.weight.requires_grad = True
            # self.hypernet.up_linear.bias.requires_grad = True

    @torch.no_grad()
    def produce_original_embeddings(
        self,
        input_ids: TensorType["batch", "seq_len"],
        attention_mask: TensorType["batch", "seq_len"],
        features: TensorType["batch", "seq_len"],
        token_type_ids: Optional[TensorType["batch", "seq_len"]] = None,
        position_ids: Optional[TensorType["batch", "seq_len"]] = None,
        head_mask: Optional[TensorType["layers", "heads"]] = None,
    ) -> TensorType["batch", "seq_len", "hidden_size"]:
        self.train(False)
        outputs = self.last_emb(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            features=features,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            include_original=False
        )

        self.train(True)
        return outputs.last_hidden_state, attention_mask

    def last_emb(self, input_ids, attention_mask, dialect_features, original_mask=None, original_embedding=None, include_original=True, **kwargs):
        """
        forward model needs to include dialect_features parameter for Trainer to not discard this feature
        """
        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask, **kwargs}
        if include_original:
            inputs["original_embedding"] = original_embedding
            inputs["original_mask"] = original_mask

        self.hypernet.down_hypernet.set_dialect_features(
            self.emb(dialect_features))
        self.hypernet.up_hypernet.set_dialect_features(
            self.emb(dialect_features))
        # self.hypernet.layernorm_w_hypernet.set_dialect_features(self.emb(dialect_features))
        # self.hypernet.layernorm_b_hypernet.set_dialect_features(self.emb(dialect_features))
        outputs = self.model(**inputs)
        return outputs

    def get_weight(self, mask):
        probs = torch.div(mask, mask.sum(1).reshape(-1, 1))
        return probs

    def emb(self, l):
        """
        PCA Embedding of linguistic attestation vector
        """
        feature = l
        if not isinstance(l, torch.Tensor):
            feature = torch.Tensor(l)
        return feature

    def forward(self, labels, input_ids, attention_mask, features, instruction_input=None, instruction_attention_mask=None, original_mask=None, original_embedding=None, include_original=False, **kwargs):
        """
        forward model needs to include features parameter for Trainer to not discard this feature
        """
        inputs = {"labels": labels, "input_ids": input_ids,
                  "attention_mask": attention_mask, **kwargs}
        if include_original:
            inputs["original_embedding"] = original_embedding
            inputs["original_mask"] = original_mask

        if not self.args.whitening:
            features = self.adap_pooler(features)

        self.model.instruction_input = instruction_input
        self.model.instruction_attention_mask = instruction_attention_mask

        self.hypernet.down_hypernet.set_features(self.emb(features))
        self.hypernet.up_hypernet.set_features(self.emb(features))
        if 'hyperlora' in self.args.name:
            self.decoder_hypernet.down_hypernet.set_features(
                self.emb(features))
            self.decoder_hypernet.up_hypernet.set_features(self.emb(features))
            self.cross_hypernet.down_hypernet.set_features(self.emb(features))
            self.cross_hypernet.up_hypernet.set_features(self.emb(features))
        if 'hyfn' in self.args.name:
            self.ffn_en_hypernet_wi.down_hypernet.set_features(
                self.emb(features))
            self.ffn_en_hypernet_wi.up_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wi.down_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wi.up_hypernet.set_features(
                self.emb(features))
            self.ffn_en_hypernet_wo.down_hypernet.set_features(
                self.emb(features))
            self.ffn_en_hypernet_wo.up_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wo.down_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wo.up_hypernet.set_features(
                self.emb(features))

        outputs = self.model(**inputs)

        return outputs

    def generate(self, input_ids, attention_mask, features=None, instruction_input=None, instruction_attention_mask=None, **kwargs):
        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask, **kwargs}

        if not self.args.whitening:
            features = self.adap_pooler(features)

        self.model.instruction_input = instruction_input
        self.model.instruction_attention_mask = instruction_attention_mask

        self.hypernet.down_hypernet.set_features(self.emb(features))
        self.hypernet.up_hypernet.set_features(self.emb(features))
        if 'hyperlora' in self.args.name:
            self.decoder_hypernet.down_hypernet.set_features(
                self.emb(features))
            self.decoder_hypernet.up_hypernet.set_features(self.emb(features))
            self.cross_hypernet.down_hypernet.set_features(self.emb(features))
            self.cross_hypernet.up_hypernet.set_features(self.emb(features))
        if 'hyfn' in self.args.name:
            self.ffn_en_hypernet_wi.down_hypernet.set_features(
                self.emb(features))
            self.ffn_en_hypernet_wi.up_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wi.down_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wi.up_hypernet.set_features(
                self.emb(features))
            self.ffn_en_hypernet_wo.down_hypernet.set_features(
                self.emb(features))
            self.ffn_en_hypernet_wo.up_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wo.down_hypernet.set_features(
                self.emb(features))
            self.ffn_de_hypernet_wo.up_hypernet.set_features(
                self.emb(features))

        return self.model.generate(**inputs)


class T5LoraWrapper(AdapterWrapper):
    def __init__(self, model, embedding_dim, weights, args):
        super().__init__(model, embedding_dim, weights, args)

    def init_hypernet(self):
        for i, l in enumerate(self.model.encoder.block):
            l = l.module if hasattr(l, "module") else l
            l.layer[0].SelfAttention.q = HyperLora(
                l.layer[0].SelfAttention.q, self.hypernet.down_hypernet, self.hypernet.up_hypernet, 2*i)
            l.layer[0].SelfAttention.v = HyperLora(
                l.layer[0].SelfAttention.v, self.hypernet.down_hypernet, self.hypernet.up_hypernet, 2*i+1)
        if 'hyperlora' in self.args.name:
            for i, l in enumerate(self.model.decoder.block):
                l = l.module if hasattr(l, "module") else l
                l.layer[0].SelfAttention.q = HyperLora(
                    l.layer[0].SelfAttention.q, self.decoder_hypernet.down_hypernet, self.decoder_hypernet.up_hypernet, 2*i)
                l.layer[0].SelfAttention.v = HyperLora(
                    l.layer[0].SelfAttention.v, self.decoder_hypernet.down_hypernet, self.decoder_hypernet.up_hypernet, 2*i+1)
                l.layer[1].EncDecAttention.q = HyperLora(
                    l.layer[1].EncDecAttention.q, self.cross_hypernet.down_hypernet, self.cross_hypernet.up_hypernet, 2*i)
                l.layer[1].EncDecAttention.v = HyperLora(
                    l.layer[1].EncDecAttention.v, self.cross_hypernet.down_hypernet, self.cross_hypernet.up_hypernet, 2*i+1)
        if 'hyfn' in self.args.name:
            for i, l in enumerate(self.model.encoder.block):
                l = l.module if hasattr(l, "module") else l
                l.layer[1].DenseReluDense.wi_0 = HyperLora(
                    l.layer[1].DenseReluDense.wi_0, self.ffn_en_hypernet_wi.down_hypernet, self.ffn_en_hypernet_wi.up_hypernet, 2*i)
                l.layer[1].DenseReluDense.wi_1 = HyperLora(
                    l.layer[1].DenseReluDense.wi_1, self.ffn_en_hypernet_wi.down_hypernet, self.ffn_en_hypernet_wi.up_hypernet, 2*i+1)
                l.layer[1].DenseReluDense.wo = HyperLora(
                    l.layer[1].DenseReluDense.wo, self.ffn_en_hypernet_wo.down_hypernet, self.ffn_en_hypernet_wo.up_hypernet, i)
            for i, l in enumerate(self.model.decoder.block):
                l = l.module if hasattr(l, "module") else l
                l.layer[2].DenseReluDense.wi_0 = HyperLora(
                    l.layer[2].DenseReluDense.wi_0, self.ffn_de_hypernet_wi.down_hypernet, self.ffn_de_hypernet_wi.up_hypernet, 2*i)
                l.layer[2].DenseReluDense.wi_1 = HyperLora(
                    l.layer[2].DenseReluDense.wi_1, self.ffn_de_hypernet_wi.down_hypernet, self.ffn_de_hypernet_wi.up_hypernet, 2*i+1)
                l.layer[2].DenseReluDense.wo = HyperLora(
                    l.layer[2].DenseReluDense.wo, self.ffn_de_hypernet_wo.down_hypernet, self.ffn_de_hypernet_wo.up_hypernet, i)
        self.freeze_params()
