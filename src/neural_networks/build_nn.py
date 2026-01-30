import argparse
import torch

from utils.gpu_setup import is_main

from configs.constants import TRANSFORMER_MODELS, MAE_MODELS


class BuildNN:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def build_nn(self, data_representation):
        nn_components = None

        if "trans" in self.args.neural_network:
            nn_components = self.prepare_transformer(data_representation)
            nn_components["find_unused_parameters"] = TRANSFORMER_MODELS[self.args.neural_network]["find_unused_parameters"]
        if "mae" in self.args.neural_network:
            nn_components = self.prepare_mae()
            nn_components["find_unused_parameters"] = MAE_MODELS[self.args.neural_network]["find_unused_parameters"]
        assert nn_components is not None, print("NN Components is None")

        d_model = nn_components["neural_network"].cfg.d_model
        add_head_first = self.args.add_task_head and self.args.ckpt_has_head

        if add_head_first:
            nn_components["neural_network"] = self.add_task_head(nn_components["neural_network"], d_model)

        if self.args.nn_ckpt:
            self.load_nn_checkpoint(nn_components, data_representation)

        if self.args.add_task_head and not self.args.ckpt_has_head:
            nn_components["neural_network"] = self.add_task_head(nn_components["neural_network"], d_model)


        return nn_components

    def prepare_transformer(self, data_representation):
        if "trans_discrete" in self.args.neural_network:
            vocab_size = data_representation.vocab_size
            if self.args.nn_ckpt:
                ckpt = torch.load(self.args.nn_ckpt, map_location="cpu", weights_only=False)
                vocab_size = ckpt["model_state_dict"]["token_emb.weight"].shape[0]
            if self.args.neural_network == "trans_discrete_decoder":
                from neural_networks.transformer.discrete.decoder import DecoderTransformerConfig, DecoderTransformer

                cfg = DecoderTransformerConfig(vocab_size=vocab_size, pad_id=self.args.pad_id, max_seq_len=self.args.bpe_symbolic_len)
                model = DecoderTransformer(cfg)
                if self.args.bfloat_16:
                    model = model.to(torch.bfloat16) # decoder only usually bfloat16
            elif self.args.neural_network == "trans_discrete_encoder":
                from neural_networks.transformer.discrete.encoder import EncoderTransformerConfig, EncoderTransformer

                cfg = EncoderTransformerConfig(vocab_size=vocab_size, pad_id=self.args.pad_id, max_seq_len=self.args.bpe_symbolic_len)
                model = EncoderTransformer(cfg)
            elif self.args.neural_network == "trans_discrete_seq2seq":
                from neural_networks.transformer.discrete.seq2seq import Seq2SeqTransformerConfig, Seq2SeqTransformer

                cfg = Seq2SeqTransformerConfig(vocab_size=vocab_size, pad_id=self.args.pad_id, max_seq_len=self.args.bpe_symbolic_len)
                model = Seq2SeqTransformer(cfg)
        elif "trans_continuous" in self.args.neural_network:
            if self.args.neural_network == "trans_continuous_nepa":
                from neural_networks.transformer.continuous.nepa import NEPAConfig, NEPATransformer

                cfg = NEPAConfig(max_seq_len=self.args.segment_len)
                model = NEPATransformer(cfg)
            elif self.args.neural_network == "trans_continuous_dit":
                from neural_networks.transformer.continuous.dit import DiTConfig, DiT

                num_steps = {"rectified_flow": 50, "ddpm": 1000}[self.args.objective]
                cfg = DiTConfig(loss_type=self.args.objective, num_steps=num_steps)
                model = DiT(cfg)
        return {"neural_network": model}

    def prepare_mae(self, ):
        if self.args.neural_network == "mae_vit":
            from neural_networks.mae.mae_vit import MAEViTConfig, MAEViT

            cfg = MAEViTConfig(patch_dim=self.args.patch_dim)
            model = MAEViT(cfg)
        return {"neural_network": model}

    def add_task_head(self, model, hidden_dim):
        if is_main():
            print(f"Adding {self.args.task} Head")
        if self.args.task == "classification":
            from neural_networks.task_heads.classification_head import ClassificationHead

            labels = self.args.batch_labels
            model = ClassificationHead(model, hidden_dim=hidden_dim, labels=labels)
        return model

    def load_nn_checkpoint(self, nn_components, data_representation):
        ckpt = torch.load(self.args.nn_ckpt, map_location="cpu", weights_only=False)

        # Use EMA weights when available and requested
        use_ema = getattr(self.args, "ema", False) and "ema_state_dict" in ckpt
        if use_ema:
            state = ckpt["ema_state_dict"]
            if is_main():
                print("Loading EMA weights from checkpoint")
        else:
            state = ckpt["model_state_dict"]

        if "trans_discrete" in self.args.neural_network:
            old_vocab = state["token_emb.weight"].shape[0]
            new_vocab = data_representation.vocab_size
            nn_components["neural_network"].load_state_dict(state, strict=True)
            if new_vocab > old_vocab:
                nn_components["neural_network"].resize_embeddings(new_vocab)
                if is_main():
                    print(f"Resized vocab from {old_vocab} to {new_vocab}")
        else:
            nn_components["neural_network"].load_state_dict(state, strict=False)
        if is_main():
            print(f"Loaded NN checkpoint from {self.args.nn_ckpt}")