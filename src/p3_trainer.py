import string
import re
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from datasets import load_metric
from transformers.trainer_callback import TrainerCallback
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
)
from losses import att_mse_loss, cos_loss
from model import T5LoraWrapper
import torch.distributed as dist


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        return control


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def trim_batch_3d(
    hidden_states,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by 0 in attention_mask"""
    # used in FiD-Pair
    keep_column_mask = attention_mask.ne(0).any(dim=0)
    return hidden_states[:, keep_column_mask, :], attention_mask[:, keep_column_mask]


def label_smoothed_nll_loss(lprobs, target, epsilon=0.1, ignore_index=-100, average="instance"):
    """From fairseq"""

    # add an extra dimension for gathering, [bsz, seqlen, 1]
    target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    pad_mask = target.eq(ignore_index)
    nll_loss.masked_fill_(pad_mask, 0.0)
    smooth_loss.masked_fill_(pad_mask, 0.0)

    eps_i = epsilon / lprobs.size(-1)

    if average == "sum":
        # return the sum of loss over all tokens in this batch
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    if average == "average":
        # return the average loss for all tokens in this batch (should be identical to output.loss)
        n_tokens = target.ne(ignore_index).sum()
        nll_loss = nll_loss.sum() / n_tokens
        smooth_loss = smooth_loss.sum() / n_tokens
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    if average == "instance_sum":
        # for each instance, compute the sum of loss over all tokens
        # return a 1d tensor, representing loss for each instance
        nll_loss = nll_loss.squeeze(-1).sum(-1)
        smooth_loss = smooth_loss.squeeze(-1).sum(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    if average == "instance":
        # for each instance, compute the per-token average loss
        # return a 1d tensor, representing loss for each instance
        token_per_instance = target.shape[-2] - pad_mask.sum(1)
        token_per_instance = token_per_instance.squeeze(-1)
        nll_loss = nll_loss.squeeze(-1).sum(-1) / token_per_instance
        smooth_loss = smooth_loss.squeeze(-1).sum(-1) / token_per_instance
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    if average is None:
        # return non-normalized, per-token loss
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss


class P3KDTrainer(Seq2SeqTrainer):

    def post_init(self, args, t_model):
        self.config = args
        self.t_model = t_model
        # self.t_model = AutoModelForSeq2SeqLM.from_pretrained(
        #     args.t_model,
        #     from_tf=bool(".ckpt" in args.t_model),
        #     # cache_dir=model_args.cache_dir,
        #     revision=args.model_revision,
        #     use_auth_token=True if args.use_auth_token else None,
        # )
        # for layer in self.t_model.modules():
        #     for _, param in layer.named_parameters():
        #         param.requires_grad = False

    def trim_batch(self, batch, pad_token_id):
        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        if len(batch) > 2:
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])
            if len(batch) > 5:
                batch[4], batch[5] = trim_batch(
                    batch[4], pad_token_id, batch[5])
        return batch

    # ideally precompute_support_hidden_states + run_model_eval should be similar to run_model
    def precompute_support_hidden_states(self, bsz, batch):
        support_input_ids = batch[0]
        support_attention_mask = batch[1]

        # max_length = max([m.sum() for m in support_attention_mask]).item()
        demo_length = support_input_ids.size(0) // bsz
        support_hidden_states, support_attention_masks = [], []

        for i in range(bsz):
            with torch.no_grad():
                attention_mask = support_attention_mask[i *
                                                        demo_length:(i+1)*demo_length]
                hidden_states = self.t_model.encoder(
                    input_ids=support_input_ids[i *
                                                demo_length:(i+1)*demo_length],
                    attention_mask=attention_mask
                ).last_hidden_state
                # support_hidden_states: k * src_len * hidden_dim
                hidden_states = hidden_states.flatten(0, 1)
                # support_hidden_states: (k * src_len) * hidden_dim
                attention_mask = attention_mask.flatten(0, 1)
                # support_attention_mask: (k * src_len)

                # remove padding positions
                hidden_states = hidden_states[attention_mask.bool(), :]
                # support_hidden_states: (new_length) * hidden_dim
                attention_mask = torch.ones(
                    1, attention_mask.sum(),
                    dtype=torch.long, device=support_attention_mask.device
                )

                support_hidden_states.append(hidden_states)
                support_attention_masks.append(attention_mask)

        return support_hidden_states, support_attention_masks

    # kd loss
    def cal_loss(self, s_logits, t_logits, temperature):
        soft_labels = F.log_softmax(
            t_logits / temperature, dim=-1, dtype=torch.float32)
        log_prob = F.log_softmax(
            s_logits / temperature, dim=-1, dtype=torch.float32)
        ori_kld_loss = (
            -torch.exp(soft_labels) * log_prob +
            torch.exp(soft_labels) * soft_labels
        )
        loss = torch.mean(torch.sum(ori_kld_loss, dim=-1))

        return loss

    def compute_kernel_bias(self, vecs, n_components=256):
        """compute kernel and bias
        vecs.shape = [num_samples, embedding_size]
        transfer:y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :n_components], -mu

    def transform_and_normalize(self, vecs, kernel=None, bias=None):
        """ normalization
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

    def preprocess_function(self, sample):
        output = self.t_model.encoder(**sample, return_dict=True)
        # shape is [batch_size, seq_len, hidden_size]
        pooled_sentence = output.last_hidden_state
        pooled_sentence = np.array(torch.mean(
            pooled_sentence, dim=1).cpu().detach().numpy())
        kernel, bias = self.compute_kernel_bias(pooled_sentence, 255)
        pooled_sentence = self.transform_and_normalize(
            pooled_sentence, kernel=kernel, bias=bias)
        sample['features'] = pooled_sentence
        return sample

    def get_features(self, inputs):
        output = self.t_model.encoder(**inputs, return_dict=True)
        # shape is [batch_size, seq_len, hidden_size]
        pooled_sentence = output.last_hidden_state
        pooled_sentence = np.array(torch.mean(
            pooled_sentence, dim=1).cpu().detach().numpy())
        kernel, bias = self.compute_kernel_bias(pooled_sentence, 255)
        pooled_sentence = self.transform_and_normalize(
            pooled_sentence, kernel=kernel, bias=bias)
        return pooled_sentence

    def query_logits_distill(self,
                             s_logits,
                             t_logits,
                             s_input_attention_mask,
                             t_input_attention_mask,
                             ):
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        # extract the s_logits
        flat_s_logits = s_logits.reshape(-1, s_logits.shape[-1]).contiguous()
        flat_t_logits = t_logits.reshape(-1, t_logits.shape[-1]).contiguous()
        flat_t_attention_mask = t_input_attention_mask.reshape(
            -1, 1).contiguous().bool()
        flat_s_attention_mask = s_input_attention_mask.reshape(
            -1, 1).contiguous().bool()
        select_s_logits = torch.masked_select(
            flat_s_logits, flat_s_attention_mask).reshape(-1, flat_s_logits.shape[-1])
        select_t_logits = torch.masked_select(
            flat_t_logits, flat_t_attention_mask).reshape(-1, flat_t_logits.shape[-1])

        kl_loss = loss_fn(
            F.log_softmax(select_s_logits/self.model_args.temperature, dim=1),
            F.softmax(select_t_logits/self.model_args.temperature, dim=1)
        )
        return kl_loss

    def cal_kl(self, logits, t_logits):
        p_s = F.log_softmax(logits / 4.0, dim=-1)
        p_t = F.softmax(t_logits / 4.0, dim=-1)
        kl_loss = (
            F.kl_div(p_s, p_t, reduction='batchmean')
        )
        return torch.sigmoid(kl_loss) * kl_loss / 2

    def cal_hd(self, hd, t_hd, context_mask=None):
        if self.config.select:
            hd = [hd[0], hd[len(hd) // 2], hd[-1]]
            t_hd = [t_hd[0], t_hd[len(t_hd) // 2], t_hd[-1]]

        loss_h = [cos_loss(
            h,
            t_h,
            context_mask if context_mask is None else context_mask.view(
                context_mask.size(0) * context_mask.size(1), -1),
        )
            for h, t_h in zip(hd, t_hd)
        ]
        return sum(loss_h) / len(loss_h)

    def cal_attn(self, attn, t_attn, context_mask=None):
        if self.config.select:
            attn = [attn[0], attn[len(attn) // 2], attn[-1]]
            t_attn = [t_attn[0], t_attn[len(t_attn) // 2], t_attn[-1]]

        loss_a = [
            att_mse_loss(a, t_a, context_mask if context_mask is None else context_mask.view(
                context_mask.size(0) * context_mask.size(1), -1))
            for a, t_a in zip(attn, t_attn)
        ]
        return sum(loss_a) / len(loss_a)

    def normalize(self, logit):
        mean = logit.mean(dim=-1, keepdims=True)
        stdv = logit.std(dim=-1, keepdims=True)
        return (logit - mean) / (1e-7 + stdv)

    def kd_loss(self, logits_student_in, logits_teacher_in, temperature, logit_stand):
        logits_student = self.normalize(
            logits_student_in) if logit_stand else logits_student_in
        logits_teacher = self.normalize(
            logits_teacher_in) if logit_stand else logits_student_in
        log_pred_student = F.log_softmax(logits_student / temperature, dim=-1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=-1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher,
                           reduction="none").sum(-1).mean()
        loss_kd *= temperature**2
        return loss_kd

    def cal_loramse(self, model, lora_A, lora_B):
        param_tensor_A, param_tensor_B = [], []
        model = model.module if hasattr(model, "module") else model
        for i, l in enumerate(model.encoder.block): # qkvo
            l = l.module if hasattr(l, "module") else l
            if 'ko' in model.config.name:
                param_tensor_A.append(l.layer[0].SelfAttention.q.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[0].SelfAttention.k.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[0].SelfAttention.v.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[0].SelfAttention.o.lora_A.unsqueeze(1))
                
                param_tensor_B.append(l.layer[0].SelfAttention.q.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[0].SelfAttention.k.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[0].SelfAttention.v.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[0].SelfAttention.o.lora_B.unsqueeze(1))
            else:
                param_tensor_A.append(l.layer[0].SelfAttention.q.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[0].SelfAttention.v.lora_A.unsqueeze(1))
                
                param_tensor_B.append(l.layer[0].SelfAttention.q.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[0].SelfAttention.v.lora_B.unsqueeze(1))
        for i, l in enumerate(model.decoder.block): # qkvo
            l = l.module if hasattr(l, "module") else l
            if 'ko' in model.config.name:
                param_tensor_A.append(l.layer[0].SelfAttention.q.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[0].SelfAttention.k.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[0].SelfAttention.v.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[0].SelfAttention.o.lora_A.unsqueeze(1))
                
                param_tensor_B.append(l.layer[0].SelfAttention.q.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[0].SelfAttention.k.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[0].SelfAttention.v.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[0].SelfAttention.o.lora_B.unsqueeze(1))
                
                # EncDecAttention
                param_tensor_A.append(l.layer[1].EncDecAttention.q.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[1].EncDecAttention.k.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[1].EncDecAttention.v.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[1].EncDecAttention.o.lora_A.unsqueeze(1))
                
                param_tensor_B.append(l.layer[1].EncDecAttention.q.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[1].EncDecAttention.k.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[1].EncDecAttention.v.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[1].EncDecAttention.o.lora_B.unsqueeze(1))
            else:
                param_tensor_A.append(l.layer[0].SelfAttention.q.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[0].SelfAttention.v.lora_A.unsqueeze(1))
                
                param_tensor_B.append(l.layer[0].SelfAttention.q.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[0].SelfAttention.v.lora_B.unsqueeze(1))
                
                # EncDecAttention
                param_tensor_A.append(l.layer[1].EncDecAttention.q.lora_A.unsqueeze(1))
                param_tensor_A.append(l.layer[1].EncDecAttention.v.lora_A.unsqueeze(1))
                
                param_tensor_B.append(l.layer[1].EncDecAttention.q.lora_B.unsqueeze(1))
                param_tensor_B.append(l.layer[1].EncDecAttention.v.lora_B.unsqueeze(1))
        
        param_tensor_A = torch.cat(param_tensor_A, dim=1)
        param_tensor_B = torch.cat(param_tensor_B, dim=1)
        
        return (F.mse_loss(param_tensor_A, lora_B) + F.mse_loss(param_tensor_B, lora_A)) * 1000
    
    def compute_loss(self, model, inputs, return_outputs=False):
        self.t_model.eval()
        model.train()

        batch = self.trim_batch(inputs, self.tokenizer.pad_token_id)
        # k * src_len
        support_input_ids, support_attention_mask = batch[0], batch[1]
        # m * src_len
        query_input_ids, query_attention_mask = batch[2], batch[3]
        # m * tar_len
        target_input_ids, target_attention_mask = batch[4], batch[5]
        features = batch[6]
    
        target_input_ids[target_input_ids ==
                         self.tokenizer.pad_token_id] = -100

        # forward pass
        bsz = query_input_ids.size(0)
        support_hidden_states, support_attention_mask = self.precompute_support_hidden_states(
            bsz, [support_input_ids, support_attention_mask])

        query_hidden_states = self.t_model.encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask).last_hidden_state
        # concat w.r.t sequence length
        max_len = max([m.sum() for m in support_attention_mask]
                      ).item() + query_input_ids.size(1)

        all_hidden_states = torch.ones((bsz, max_len, query_hidden_states.size(-1)), dtype=torch.bfloat16, device=query_input_ids.device) * 0
        all_attention_mask = torch.ones((bsz, max_len), dtype=torch.long, device=query_input_ids.device) * 0
        for i in range(bsz):
            l = support_attention_mask[i].sum().item() + query_input_ids.size(1)
            all_hidden_states[i, :l] = torch.cat((support_hidden_states[i], query_hidden_states[i]))
            all_attention_mask[i, :l] = torch.cat((support_attention_mask[i].squeeze(0), query_attention_mask[i]))

        output_hidden_states = False
        output_attentions = False
        if self.config.use_hd:
            output_hidden_states = True
        if self.config.use_attn:
            output_attentions = True
        with torch.no_grad():
            t_outputs = self.t_model(
                attention_mask=all_attention_mask,
                encoder_outputs=[all_hidden_states],
                labels=target_input_ids,
                decoder_attention_mask=target_attention_mask,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states)
        # t_loss = t_outputs.get("loss")
        t_logits = t_outputs.get("logits")

        outputs = model(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            labels=target_input_ids,
            decoder_attention_mask=target_attention_mask,
            features=features,
            use_cache=False,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        loss = outputs.get("loss")
        logits = outputs.get("logits")
        if self.config.use_ce:
            loss = (
                self.config.alpha_kd *
                self.cal_loss(logits, t_logits, self.config.temperature)
                + (1 - self.config.alpha_kd) * loss
            )
        if self.config.use_kl:
            #     loss = self.cal_kl(logits, t_logits) * 0.1 + loss * 0.8
            # if self.config.logit_stand:
            loss += self.kd_loss(logits, t_logits,
                                 self.config.temperature, self.config.logit_stand) * 5
        if self.config.use_hd:
            loss += self.cal_hd(outputs.get("encoder_hidden_states"),
                                t_outputs.get("encoder_hidden_states"))
            loss += self.cal_hd(outputs.get("decoder_hidden_states"),
                                t_outputs.get("decoder_hidden_states")) * 10
        if self.config.use_attn:
            loss += self.cal_attn(outputs.get("encoder_attentions"),
                                  t_outputs.get("encoder_attentions")) * 1000
            loss += self.cal_attn(outputs.get("decoder_attentions"),
                                  t_outputs.get("decoder_attentions")) * 10
            loss = loss.to(torch.bfloat16)
        if self.config.loramse:
            lora_A, lora_B = batch[7].to(torch.bfloat16), batch[8].to(torch.bfloat16)
            loss_lora = self.cal_loramse(model, lora_A, lora_B)
            loss += torch.sigmoid(loss_lora) * loss_lora
        # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    # rewrite the evaluation loop, with customized call to compute_metrics
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        self._max_length = 64
        self._num_beams = 4
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # loss_list, metadata = [], []
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)

            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs[2:7], prediction_loss_only, ignore_keys=ignore_keys)

            # loss_list += loss.float().cpu().detach().numpy().tolist()

            if is_torch_tpu_available():
                xm.mark_step()

            # dist.barrier()
            # world_size = dist.get_world_size()
            # metadata_gather = [None for _ in range(world_size)]
            # loss_list_gather = [None for _ in range(world_size)]
            # dist.all_gather_object(metadata_gather, metadata)
            # dist.all_gather_object(loss_list_gather, loss_list)
            # for i in range(world_size):
            #     metadatas.extend(metadata_gather[i])
            #     loss_lists.extend(loss_list_gather[i])
            
            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss)
                losses_host = losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0)
            # if labels is not None:
            #     labels = self._pad_across_processes(labels)
            #     labels = self._nested_gather(labels)
            #     labels_host = labels if labels_host is None else nested_concat(
            #         labels_host, labels, padding_index=-100)
            # if logits is not None:
            #     logits = self._pad_across_processes(logits)
            #     logits = self._nested_gather(logits)
            #     if self.preprocess_logits_for_metrics is not None:
            #         logits = self.preprocess_logits_for_metrics(logits, labels)
            #     preds_host = logits if preds_host is None else nested_concat(
            #         preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # print(len(self._nested_gather(loss.float().cpu().detach().numpy().tolist())))
            # print(len(self._nested_gather(metadata)))

            # # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None
                
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)
        
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # print(len(metadata))
        # print(len(loss_list))
        # print(len(eval_dataset))
        # print(metadatas[-3:])
        # print(eval_dataset[-3:])
        # print(num_samples)
        start_idx = 0
        metadata = []
        for i in range(len(eval_dataset)):
            dp = eval_dataset[i]['Instance']['options']
            n_options = len(dp)
            metadata.append({
                "indices": list(range(start_idx, start_idx + n_options)),
                "options": dp,
            })
            start_idx += n_options
        # print(eval_dataset[-1])

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        metrics = self.compute_metrics(
            dataset=eval_dataset, metadata=metadata, preds=np.array(all_losses), save_prefix=metric_key_prefix)
        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        #     metrics = self.compute_metrics(
        #         dataset=eval_dataset, metadata=metadata, preds=np.array(loss_list), save_prefix=metric_key_prefix)
        # else:
        #     metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        batch = self.trim_batch(inputs, self.tokenizer.pad_token_id)
        input_ids, attention_mask = batch[0], batch[1]
        decoder_input_ids, decoder_attention_mask = batch[2], batch[3]
        features = batch[4]
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "features": features
        }

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names

        with torch.no_grad():
            output = self.model(
                **inputs,
                use_cache=False
            )

        # rank_classification
        lprobs = F.log_softmax(output.logits, dim=-1)
        loss, _ = label_smoothed_nll_loss(
            lprobs, inputs.get('labels', None),
            epsilon=0.0,
            # epsilon=0.1 if is_training else 0.0,
            ignore_index=self.tokenizer.pad_token_id,
            average='instance',  # by default it's per-instance token-avg loss
        )

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, lprobs, labels)


class P3Trainer(Seq2SeqTrainer):

    def trim_batch(self, batch, pad_token_id):
        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        if len(batch) == 5:
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])
        return batch

    def compute_loss(self, model, inputs, return_outputs=False):
        batch = self.trim_batch(inputs, self.tokenizer.pad_token_id)
        input_ids, attention_mask = batch[0], batch[1]
        decoder_input_ids, decoder_attention_mask = batch[2], batch[3]

        decoder_input_ids[decoder_input_ids ==
                          self.tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False
        )

        # rank_classification
        # lprobs = F.log_softmax(outputs.logits, dim=-1)
        # loss, _ = label_smoothed_nll_loss(
        #     lprobs, decoder_input_ids,
        #     epsilon=0.0,
        #     # epsilon=0.1 if is_training else 0.0,
        #     ignore_index=self.tokenizer.pad_token_id,
        #     average="instance", # by default it's per-instance token-avg loss
        # )
        loss = outputs.loss

        # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    # rewrite the evaluation loop, with customized call to compute_metrics
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        self._max_length = 64
        self._num_beams = 4
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # loss_list, metadata = [], []
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)

            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # loss_list += loss.float().cpu().detach().numpy().tolist()

            if is_torch_tpu_available():
                xm.mark_step()

            # dist.barrier()
            # world_size = dist.get_world_size()
            # metadata_gather = [None for _ in range(world_size)]
            # loss_list_gather = [None for _ in range(world_size)]
            # dist.all_gather_object(metadata_gather, metadata)
            # dist.all_gather_object(loss_list_gather, loss_list)
            # for i in range(world_size):
            #     metadatas.extend(metadata_gather[i])
            #     loss_lists.extend(loss_list_gather[i])
            
            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss)
                losses_host = losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0)
            # if labels is not None:
            #     labels = self._pad_across_processes(labels)
            #     labels = self._nested_gather(labels)
            #     labels_host = labels if labels_host is None else nested_concat(
            #         labels_host, labels, padding_index=-100)
            # if logits is not None:
            #     logits = self._pad_across_processes(logits)
            #     logits = self._nested_gather(logits)
            #     if self.preprocess_logits_for_metrics is not None:
            #         logits = self.preprocess_logits_for_metrics(logits, labels)
            #     preds_host = logits if preds_host is None else nested_concat(
            #         preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # print(len(self._nested_gather(loss.float().cpu().detach().numpy().tolist())))
            # print(len(self._nested_gather(metadata)))

            # # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None
                
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)
        
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # print(len(metadata))
        # print(len(loss_list))
        # print(len(eval_dataset))
        # print(metadatas[-3:])
        # print(eval_dataset[-3:])
        # print(num_samples)
        start_idx = 0
        metadata = []
        for i in range(len(eval_dataset)):
            dp = eval_dataset[i]['Instance']['options']
            n_options = len(dp)
            metadata.append({
                "indices": list(range(start_idx, start_idx + n_options)),
                "options": dp,
            })
            start_idx += n_options
        # print(eval_dataset[-1])

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        metrics = self.compute_metrics(
            dataset=eval_dataset, metadata=metadata, preds=np.array(all_losses), save_prefix=metric_key_prefix)
        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        #     metrics = self.compute_metrics(
        #         dataset=eval_dataset, metadata=metadata, preds=np.array(loss_list), save_prefix=metric_key_prefix)
        # else:
        #     metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        batch = self.trim_batch(inputs, self.tokenizer.pad_token_id)
        input_ids, attention_mask = batch[0], batch[1]
        decoder_input_ids, decoder_attention_mask = batch[2], batch[3]
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names

        with torch.no_grad():
            output = self.model(
                **inputs,
                use_cache=False
            )

        # rank_classification
        lprobs = F.log_softmax(output.logits, dim=-1)
        loss, _ = label_smoothed_nll_loss(
            lprobs, inputs.get('labels', None),
            epsilon=0.0,
            # epsilon=0.1 if is_training else 0.0,
            ignore_index=self.tokenizer.pad_token_id,
            average='instance',  # by default it's per-instance token-avg loss
        )

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, lprobs, labels)
