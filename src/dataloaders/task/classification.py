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
        if self.args.neural_network in ("mlae", "mtae", "st_mem"):
            out = {"series": inputs.astype(np.float32)}
        elif self.args.objective in ("rectified_flow", "ddpm"):
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
        # print("CLASS TOKENS", class_tokens)
        class_ids = [getattr(self.args, tok) for tok in class_tokens]
        # print("CLASS IDS", class_ids)
        inputs = inputs[:-1]
        
        # [eos, c1, eos, c2, eos, ...]
        class_seq = [self.args.eos_id]
        for cid in class_ids:
            class_seq.extend([cid, self.args.eos_id])
        class_seq = np.array(class_seq)
        
        inputs = inputs[:max_len - len(class_seq)]

        if "train" in self.args.mode:
            inputs = np.concatenate([inputs, class_seq])
            
            if self.args.objective == "autoregressive":
                num_to_mask = len(inputs) - 3
                labels = np.concatenate([
                    np.full(num_to_mask, fill_value=-100),
                    inputs[-3:]
                ])
            else:
                labels = inputs
        else:
            # print("INPUTS SHAPE", inputs.shape)
            # print("INPUTS LAST 10", inputs[-10:])
            if len(class_ids) > 1:
                inputs = np.concatenate([inputs, class_seq[:-2]])
            labels = class_seq[-3:]

        inputs = torch.as_tensor(inputs, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long)

        # print("INPUTS SHAPE", inputs.shape)
        # print("INPUTS LAST 10", inputs[-10:])
        # print("LABELS SHAPE", labels.shape)
        # print("LABELS LAST 10", labels[-10:])
        # input()

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