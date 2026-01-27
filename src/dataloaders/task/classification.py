import numpy as np
import torch

from configs.constants import BATCH_LABEL_CATS

class Classification:
    def __init__(self, args):
        self.args = args

    def __call__(self, transformed_data):
        if self.args.data_representation == "bpe_symbolic":
            return self.bpe_symbolic(transformed_data)
        elif self.args.data_representation == "signal":
            return self.signal(transformed_data)

    def signal(self, transformed_data):
        inputs = np.asarray(transformed_data["transformed_data"])
        labels = self.get_labels(transformed_data)
        if self.args.objective in ("rectified_flow", "ddpm"):
            padding_mask = transformed_data.get("padding_mask")
            out = {
                "signal": inputs,
                "padding_mask": padding_mask,
            }
        elif self.args.objective in ("mae"):
            # Each lead is a patch and we mask out 75% of them (9 leads)
            num_masked = int(self.args.num_patches * 0.75)
            perm = np.random.permutation(self.args.num_patches)
            visible_mask = np.ones(self.args.num_patches, dtype = np.float32)
            visible_mask[perm[:num_masked]] = 0
            out = {
                "patches": inputs.astype(np.float32),
                "visible_mask": visible_mask,
            }
        out.update({k: torch.as_tensor(v, dtype=torch.long) for k, v in labels.items()})
        return out

    def bpe_symbolic(self, transformed_data):
        inputs = np.asarray(transformed_data["transformed_data"])
        max_len = self.args.bpe_symbolic_len
        class_tokens = self.create_batch_tokens(transformed_data)
        class_ids = np.array([getattr(self.args, tok) for tok in class_tokens])
        inputs = inputs[:-1]
        if "train" in self.args.mode:
            inputs = inputs[: max_len - len(class_ids) - 2]
            inputs = np.concatenate([inputs, [self.args.eos_id], class_ids, [self.args.eos_id]])
            if self.args.objective == "autoregressive":
                num_masked = len(inputs) - len(class_ids) - 2
                labels = np.concatenate([np.full(num_masked, fill_value=-100), [self.args.eos_id], class_ids, [self.args.eos_id]])
        else:
            labels = np.concatenate([[self.args.eos_id], class_ids, [self.args.eos_id]])
        inputs = torch.as_tensor(inputs, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long) if labels is not None else None
        if self.args.neural_network == "trans_discrete_decoder":
            return {"tgt_ids": inputs, "labels": labels}
        elif self.args.neural_network == "trans_discrete_encoder":
            return {"src_ids": inputs, "labels": labels}
        

    def create_batch_tokens(
        self,
        transformed_data,
    ):
        tokens = []
        for label in self.args.batch_labels:
            if label in transformed_data:
                tokens.append(f"{label}_{int(transformed_data[label])}")
        return tokens

    def get_labels(self, transformed_data):
        return {k: transformed_data[k] for k in transformed_data.keys() & BATCH_LABEL_CATS.keys()}