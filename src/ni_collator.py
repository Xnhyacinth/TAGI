import logging
import random
import string
import torch
from transformers.data.data_collator import *
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForNI:

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
    data_map: dict = None

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        if not self.args.whitening:
            sources = []
            if self.kd:
                prefixs, instances, s_sources = [], [], []
                lora_A_params, lora_B_params = [], []
            for instance in batch:
                if self.tk_instruct:
                    all_valid_encodings = [
                        # instruction only
                        {"add_task_name": False, "add_task_definition": True,
                            "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False},
                        # example only
                        {"add_task_name": False, "add_task_definition": False,
                            "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False},
                        # instruction + pos examples
                        {"add_task_name": False, "add_task_definition": True,
                            "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False},
                        # instruction + pos examples + neg examples
                        {"add_task_name": False, "add_task_definition": True,
                            "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
                        # instruction + pos (w. explanation)
                        {"add_task_name": False, "add_task_definition": True,
                            "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": True},
                    ]
                    encoding_schema = random.choice(all_valid_encodings)
                    add_task_name = encoding_schema["add_task_name"]
                    add_task_definition = encoding_schema["add_task_definition"]
                    num_pos_examples = encoding_schema["num_pos_examples"]
                    num_neg_examples = encoding_schema["num_neg_examples"]
                    add_explanation = encoding_schema["add_explanation"]
                else:
                    add_task_name = self.add_task_name
                    add_task_definition = self.add_task_definition
                    num_pos_examples = self.num_pos_examples
                    num_neg_examples = self.num_neg_examples
                    add_explanation = self.add_explanation

                task_input = ""
                # add the input first.
                task_input += "Now complete the following example -\n"
                task_input += f"Input: {instance['Instance']['input'].strip()}"
                if not task_input[-1] in string.punctuation:
                    task_input += "."
                task_input += "\n"
                task_input += "Output: "

                task_name = ""
                if add_task_name:
                    task_name += instance["Task"] + ". "

                definition = ""
                if add_task_definition:
                    if isinstance(instance["Definition"], list):
                        # TODO: should we use <Definition>?
                        definition = "Definition: " + \
                            instance["Definition"][0].strip()
                    else:
                        definition = "Definition: " + \
                            instance["Definition"].strip()
                    if not definition[-1] in string.punctuation:
                        definition += "."
                    definition += "\n\n"

                # try to add positive examples.
                pos_examples = []
                for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                    pos_example_str = f" Positive Example {idx+1} -\n"
                    pos_example_str += f"Input: {pos_example['input'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                    pos_example_str += f" Output: {pos_example['output'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                    if add_explanation and "explanation" in pos_example:
                        pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                        if not pos_example_str[-1] in string.punctuation:
                            pos_example_str += "."
                        pos_example_str += "\n"
                    pos_example_str += "\n"
                    if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                        pos_examples.append(pos_example_str)
                    else:
                        # d = self.max_source_length - len(self.tokenizer(definition + " ".join(pos_examples) + task_input)["input_ids"])
                        # tokenized_pos_example_str = self.tokenizer(pos_example_str)["input_ids"]
                        # pos_examples.append(
                        #     self.tokenizer.decode(tokenized_pos_example_str[:d // 2], skip_special_tokens=True) +
                        #     self.tokenizer.decode(tokenized_pos_example_str[-d // 2:], skip_special_tokens=True)
                        #     )
                        break

                # try to add negative examples.
                neg_examples = []
                for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                    neg_example_str = f" Negative Example {idx+1} -\n"
                    neg_example_str += f"Input: {neg_example['input'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                    neg_example_str += f" Output: {neg_example['output'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                    if add_explanation and "explanation" in neg_example:
                        neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                        if not neg_example_str[-1] in string.punctuation:
                            neg_example_str += "."
                        neg_example_str += "\n"
                    neg_example_str += "\n"
                    if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                        neg_examples.append(neg_example_str)
                    else:
                        break

                source = task_name + definition + \
                    "".join(pos_examples) + "".join(neg_examples) + task_input
                tokenized_source = self.tokenizer(source)["input_ids"]
                if len(tokenized_source) <= self.max_source_length:
                    sources.append(source)
                else:
                    sources.append(self.tokenizer.decode(
                        tokenized_source[:self.max_source_length], skip_special_tokens=True))

                if self.student_input:
                    # s_source
                    s_source = task_name + definition + \
                        "".join(
                            pos_examples[:self.args.s_num_pos_examples]) + task_input
                    tokenized_s_source = self.tokenizer(s_source)["input_ids"]
                    if len(tokenized_s_source) <= self.max_source_length:
                        s_sources.append(s_source)
                    else:
                        s_sources.append(self.tokenizer.decode(
                            tokenized_s_source[:self.max_source_length], skip_special_tokens=True))

                if self.kd:
                    # prefix
                    prefix = task_name + definition + \
                        "".join(
                            pos_examples[:self.args.s_num_pos_examples]) + "".join(neg_examples)
                    prefixs.append(prefix)

                if self.args.custom_model:
                    # instance
                    instances.append(definition + task_input)
                    
                if self.args.loramse and instance['Categories'][0] in self.data_map.values():
                    if 'ko' in self.args.name:
                        lora_A_params.append(self.lora_dict[instance['Categories'][0]]['param_tensor_A'])
                        lora_B_params.append(self.lora_dict[instance['Categories'][0]]['param_tensor_B'])
                    else:
                        lora_A_params.append(self.lora_dict[instance['Categories'][0]]['param_tensor_qv_A'])
                        lora_B_params.append(self.lora_dict[instance['Categories'][0]]['param_tensor_qv_B'])

        else:
            sources, features, instances, s_sources, instruction_inputs, attention_masks = [
            ], [], [], [], [], []
            for instance in batch:
                sources.append(instance["source"])
                features.append(self.task_features[instance['Task']])
                if self.student_input:
                    s_sources.append(instance["s_source"])
                if self.args.custom_model:
                    instances.append(instance["instance"])
                    instruction_inputs.append(
                        self.instruction_inputs[instance['Task']])
                    attention_masks.append(
                        self.attention_masks[instance['Task']])
        if self.text_only:
            t_model_inputs = {"inputs": sources}
        else:
            t_model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            if self.text_only:
                t_model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                t_model_inputs["labels"] = labels["input_ids"].masked_fill(
                    ~label_mask, self.label_pad_token_id)
        else:
            t_model_inputs["labels"] = None

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=t_model_inputs["labels"])
            t_model_inputs["decoder_input_ids"] = decoder_input_ids
        if not self.kd:
            return t_model_inputs
        else:
            if not self.args.whitening:
                with self.tokenizer.as_target_tokenizer():
                    prefixs_inputs = self.tokenizer(
                        prefixs,
                        max_length=self.max_source_length,
                        padding="max_length" if self.args.prefix_length > 0 and not 'gpt' in self.args.name else self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                    if self.args.hyperencoder:
                        features = prefixs_inputs
                    else:
                        prefixs_inputs = prefixs_inputs.to(self.model.device)
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

            if self.args.custom_model:
                if self.text_only:
                    model_inputs = {"inputs": instances}
                else:
                    with self.tokenizer.as_target_tokenizer():
                        model_inputs = self.tokenizer(
                            instances,
                            max_length=self.max_source_length,
                            padding=self.padding,
                            return_tensors=self.return_tensors,
                            truncation=True,
                            pad_to_multiple_of=self.pad_to_multiple_of
                        )
                model_inputs["labels"] = t_model_inputs["labels"]
                if "decoder_input_ids" in t_model_inputs.keys():
                    model_inputs["decoder_input_ids"] = decoder_input_ids
            elif self.student_input:
                if self.text_only:
                    model_inputs = {"inputs": s_sources}
                else:
                    model_inputs = self.tokenizer(
                        s_sources,
                        max_length=self.max_source_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                model_inputs["labels"] = t_model_inputs["labels"]
                if "decoder_input_ids" in t_model_inputs.keys():
                    model_inputs["decoder_input_ids"] = decoder_input_ids
            else:
                model_inputs = t_model_inputs.copy()
            if self.args.hyperencoder:
                model_inputs["features"] = features
            else:
                model_inputs["features"] = torch.Tensor(features)
                if self.args.prefix_length > 0 or self.args.custom_model:
                    model_inputs["instruction_input"] = torch.Tensor(
                        instruction_inputs)
                    model_inputs["instruction_attention_mask"] = torch.Tensor(
                        attention_masks)
                if "gpt" in self.args.name:
                    model_inputs["instruction_input"] = prefixs_inputs.to(
                        'cpu')
            if self.args.loramse and len(lora_A_params) > 0:
                model_inputs["lora_A"] = torch.Tensor(lora_A_params)
                model_inputs["lora_B"] = torch.Tensor(lora_B_params)
                
            return t_model_inputs, model_inputs
