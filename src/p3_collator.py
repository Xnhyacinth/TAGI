import logging
import random
import string
import torch
from transformers.data.data_collator import *
from t0_config import eval

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForP3:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool = False
    kd: bool = False
    task_features: dict = None
    instruction_inputs: dict = None
    attention_masks: dict = None
    args: dict = None
    student_input: bool = False
    lora_dict: dict = None

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        def pad_tokens(lst, max_len, pad_id=None):
            # everything is padding token in the beginning
            if pad_id is None:
                pad_id = self.tokenizer.pad_token_id

            tensor = torch.ones(len(lst), max_len, dtype=torch.long) * pad_id
            # then fill each example into this big tensor
            for i, item in enumerate(lst):
                if len(item) > max_len:
                    tensor[i, :] = torch.LongTensor(item[:max_len])
                else:
                    tensor[i, :len(item)] = torch.LongTensor(item)
            return tensor

        def concat_input_and_output(_input, _output, input_prefix, output_prefix, max_input_length, max_output_length):
            return input_prefix + _input[:max_input_length-len(input_prefix)] \
                + output_prefix + \
                _output[:max_output_length-len(output_prefix)]

        if not self.args.whitening:
            sources, targets, options = [], [], []
            if self.kd:
                batch_concat_ids, batch_concat_attention_mask = [], []
                lora_A_params, lora_B_params = [], []
            for instance in batch:
                # task_input = ""
                # # add the input first.
                # task_input += "Now complete the following example -\n"
                # task_input += f"Input: "
                # task_output = "\n"
                # task_output += "Output: "
                # input_prefix = self.tokenizer(task_input, add_special_tokens=False)["input_ids"]
                # output_prefix = self.tokenizer(task_output, add_special_tokens=False)["input_ids"]

                if instance['Task'] in eval:
                    inputs = [instance['Instance']["input"]] * \
                        len(instance['Instance']["options"])
                    outputs = instance['Instance']["options"]
                    # source = self.tokenizer(instance['Instance']['input'])["input_ids"]
                    # targets.append(self.tokenizer(instance['Instance']['output'])["input_ids"])
                    tokenized_input = self.tokenizer.batch_encode_plus(inputs,
                                                                       padding='max_length',
                                                                       truncation=True,
                                                                       max_length=self.max_source_length,
                                                                       add_special_tokens=False)
                    tokenized_output = self.tokenizer.batch_encode_plus(outputs,
                                                                        padding='max_length',
                                                                        truncation=True,
                                                                        max_length=self.max_source_length,
                                                                        add_special_tokens=True)
                    sources += tokenized_input["input_ids"]
                    targets += tokenized_output["input_ids"]

                    options.append(instance['Instance']['options'])
                else:
                    source = instance["Instance"]["input_tokenized"]
                    targets.append(instance["Instance"]["output_tokenized"])

                    if len(source) <= self.max_source_length:
                        sources.append(source)
                    else:
                        sources.append(source[:self.max_source_length])

                    if self.student_input:
                        # s_source
                        pass

                    if self.kd:
                        # prefix
                        input_prefix = self.tokenizer(
                            "# Input #", add_special_tokens=False)["input_ids"]
                        output_prefix = self.tokenizer(
                            " # Output #", add_special_tokens=False)["input_ids"]

                        all_input_ids, all_target_ids = [], []
                        for idx, example in enumerate(instance["Examples"]):
                            all_input_ids.append(example["input"])
                            all_target_ids.append(example['output'])
                        concat_ids = [
                            concat_input_and_output(
                                _input, _output, input_prefix, output_prefix, self.max_source_length, self.max_target_length)
                            for _input, _output in zip(all_input_ids, all_target_ids)
                        ]
                        concat_ids = pad_tokens(
                            concat_ids, max_len=self.max_source_length+self.max_target_length)
                        concat_attention_mask = concat_ids.ne(
                            self.tokenizer.pad_token_id).long()

                        batch_concat_ids.append(concat_ids)
                        batch_concat_attention_mask.append(
                            concat_attention_mask)
                    
                    if self.args.loramse and instance['Categories'] in self.lora_dict.keys():
                        if 'ko' in self.args.name:
                            lora_A_params.append(self.lora_dict[instance['Categories']]['param_tensor_A'])
                            lora_B_params.append(self.lora_dict[instance['Categories']]['param_tensor_B'])
                        else:
                            lora_A_params.append(self.lora_dict[instance['Categories']]['param_tensor_qv_A'])
                            lora_B_params.append(self.lora_dict[instance['Categories']]['param_tensor_qv_B'])

                    if self.args.custom_model:
                        # instance
                        pass

        else:
            pass

        input_ids = pad_tokens(sources, max_len=self.max_source_length)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        decoder_input_ids = pad_tokens(targets, max_len=self.max_target_length)
        decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).long()

        if self.kd:
            if instance['Task'] not in eval:
                concat_ids = torch.cat(batch_concat_ids, dim=0)
                concat_attention_mask = torch.cat(
                    batch_concat_attention_mask, dim=0)
            else:
                concat_ids, concat_attention_mask = None, None
            prefixs_inputs = {
                "input_ids": input_ids.to(self.model.device),
                "attention_mask": attention_mask.to(self.model.device)
            }
            self.model.eval()
            with torch.no_grad():
                hidden_states = self.model.encoder(
                    **prefixs_inputs, return_dict=True, output_hidden_states=True).hidden_states

                if self.args.pooling == 'first_last_avg':
                    pooled_sentence = (
                        hidden_states[-1] + hidden_states[1])
                elif self.args.pooling == 'last_avg':
                    pooled_sentence = (hidden_states[-1])
                elif self.args.pooling == 'last2avg':
                    pooled_sentence = (
                        hidden_states[-1] + hidden_states[-2])
                else:
                    raise Exception(
                        "unknown pooling {}".format(self.args.pooling))

            if self.args.custom_model or self.args.prefix_length > 0:
                instruction_inputs = hidden_states[-1].cpu()
                attention_masks = prefixs_inputs['attention_mask'].cpu(
                )
            features = pooled_sentence.mean(dim=1).cpu()
            features = torch.Tensor(features)

            if self.args.loramse and len(lora_A_params) > 0:
                if len(lora_A_params) == input_ids.size(0):
                    lora_A_params = torch.Tensor(lora_A_params)
                    lora_B_params = torch.Tensor(lora_B_params)
                else:
                    for instance in batch:
                        if instance['Categories'] not in self.lora_dict.keys():
                            print(instance['Categories'])
            
            return concat_ids, concat_attention_mask, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, features, lora_A_params, lora_B_params
        else:
            return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask
