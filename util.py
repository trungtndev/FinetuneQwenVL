import os
from typing import List, Optional, Tuple, Union
import pytorch_lightning as pl
import torch
from torchmetrics import Metric

class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[str], indices: List[str]):
        for pred, truth in zip(indices_hat, indices):
            is_same = pred == truth

            if is_same:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate

class HFCheckpoint(pl.Callback):
    def __init__(
            self,
            output_dir,
            monitor="val_loss",
            mode="min",
            save_best=True,
            save_last=True,
            save_interval_epochs=1,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.save_interval_epochs = save_interval_epochs

        # Khởi tạo điểm số tốt nhất ban đầu (Vô cực)
        self.best_score = float('inf') if mode == "min" else float('-inf')

    def _save_hf_format(self, trainer, pl_module, folder_name):
        if trainer.is_global_zero:  # Chỉ lưu ở process chính
            save_path = os.path.join(self.output_dir, folder_name)
            os.makedirs(save_path, exist_ok=True)

            pl_module.model.save_pretrained(save_path)
            pl_module.processor.save_pretrained(save_path)
            return save_path

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if self.save_last:
            self._save_hf_format(trainer, pl_module, "last")

        # if self.save_interval_epochs > 0:
        #     if (epoch) % self.save_interval_epochs == 0:
        #         folder_name = f"epoch_{epoch}"
        #         path = self._save_hf_format(trainer, pl_module, folder_name)
        #         print(f"\n--> [Checkpoint] Saved history: {path}")

    def on_validation_end(self, trainer, pl_module):
        if not self.save_best:
            return

        metrics = trainer.callback_metrics

        if self.monitor in metrics:
            current_score = metrics[self.monitor].item()
            is_best = False
            if self.mode == "min":
                if current_score < self.best_score:
                    is_best = True
            else:  # mode max
                if current_score > self.best_score:
                    is_best = True

            if is_best:
                previous_best = self.best_score
                self.best_score = current_score
                path = self._save_hf_format(trainer, pl_module, "best")
                print(f"\n--> [Checkpoint] NEW BEST MODEL! ({self.monitor}: {previous_best:.4f} -> {current_score:.4f})\n")