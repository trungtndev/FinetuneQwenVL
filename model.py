import zipfile

from torch import optim
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLConfig
from PIL import Image
import torch
from peft import LoraConfig, get_peft_model
import pytorch_lightning as pl
from util import ExpRateRecorder


class LitQwen3VL(pl.LightningModule):
    def __init__(
            self,
            train_config,
            model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct",
    ):
        super().__init__()
        self.save_hyperparameters()

        # self.model = Qwen3VLForConditionalGeneration.from_pretrained(
        #     model_name_or_path,
        #     trust_remote_code=True
        # )
        # self.processor = AutoProcessor.from_pretrained(
        #     model_name_or_path,
        #     trust_remote_code=True
        # )
        #
        # self.exprate_recorder = ExpRateRecorder()
        #
        # lora_config = LoraConfig(
        #     r=16,
        #     lora_alpha=32,  # Hệ số scale
        #     target_modules=[
        #         # Encoder
        #         # "proj",
        #         "qkv",
        #         # "linear_fc1", "linear_fc2",
        #         # Decoder
        #         "q_proj", "k_proj", "v_proj", "o_proj",
        #         # "gate_proj", "up_proj", "down_proj"
        #     ],
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type="CAUSAL_LM"
        # )
        # self.model = get_peft_model(self.model, lora_config)
        # self.model.print_trainable_parameters()

        self.model = Qwen3VLForConditionalGeneration(
            config=Qwen3VLConfig(
                text_config={
                    "hidden_size": 128,
                    "intermediate_size": 512,
                    "num_hidden_layers": 4,

                    "head_dim": 32,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8,
                    "rope_scaling": {
                        "mrope_interleaved": True,
                        "mrope_section": [
                            24,
                            20,
                            20
                        ],
                        "rope_theta": 5000000,
                        "rope_type": "default"
                    },
                },

                vision_config={
                    'depth': 5,
                    "deepstack_visual_indexes": [1, 2, 3, 4],
                    "hidden_size": 128,
                    "intermediate_size": 512,
                    "num_heads": 8,
                    "out_hidden_size": 128,
                }
            )
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            trust_remote_code=True
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, _):
        model_inputs = self._prepare_batch(batch, is_inference=False)
        bz = model_inputs["input_ids"].size(0)
        outputs = self(**model_inputs)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bz)
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, _):
        # Tính loss bình thường
        model_inputs = self._prepare_batch(batch, is_inference=False)
        bz = model_inputs["input_ids"].size(0)

        outputs = self(**model_inputs)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=bz)
        # ====== Inference để tính ExpRate ======
        model_inputs_gen = self._prepare_batch(batch, is_inference=True)

        generated_ids = self.model.generate(
            **model_inputs_gen,
            max_new_tokens=256, num_beams=5, early_stopping=True, length_penalty=1.0
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs_gen["input_ids"], generated_ids)
        ]

        preds = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        preds = [p.strip().strip('$').strip() for p in preds]
        self.exprate_recorder(preds, batch["captions"])

        self.log(
            "val_ExpRate", self.exprate_recorder,
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=bz
        )

    @torch.inference_mode()
    def test_step(self, batch, _):
        model_inputs = self._prepare_batch(batch, is_inference=True)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256, num_beams=5, early_stopping=True, length_penalty=1.0
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
        ]

        preds = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
        preds = [p.strip().strip('$').strip() for p in preds]

        self.exprate_recorder(preds, batch["captions"])
        return batch["fnames"], preds

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.train_config.learning_rate,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.train_config.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _prepare_batch(self, batch_data, is_inference=False):
        """Hàm biến data thô thành Tensor ngay trên GPU"""
        images = batch_data["images"]
        captions = batch_data["captions"]

        texts = []
        for img, caption in zip(images, captions):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Convert the handwritten math expression image to LaTeX."},
                    ]
                }
            ]

            if not is_inference:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"$${caption}$$"}]
                })

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=is_inference
            )
            texts.append(text)

        # 1. Gọi processor tạo tensor (Lúc này đang ở trong Step)
        batch_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )

        # 2. Đẩy data lên thiết bị hiện tại (GPU)
        batch_inputs = batch_inputs.to(self.device)

        # 3. Tạo labels tính Loss cho lúc Train/Val
        if not is_inference:
            labels = batch_inputs["input_ids"].clone()
            for i in range(labels.shape[0]):
                labels[i, batch_inputs["attention_mask"][i] == 0] = -100
                label_list = labels[i].tolist()

                start_tags = [idx for idx, val in enumerate(label_list) if val == 151644]
                if len(start_tags) >= 2:
                    assistant_start_idx = start_tags[-1]
                    labels[i, :assistant_start_idx + 3] = -100
                else:
                    labels[i, :] = -100

            batch_inputs["labels"] = labels

        return batch_inputs
