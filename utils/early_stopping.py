import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=False, save_path=None):
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): 'min' or 'max' — whether lower or higher is better.
            verbose (bool): Print message when early stopping triggers.
            save_path (str): If set, saves best model to this path.
        """
        assert mode in ['min', 'max'], "mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path

        self.best_score = None
        self.counter = 0
        self.should_stop = False

        if mode == 'min':
            self.monitor_op = lambda curr, best: curr < best - min_delta
        else:  # max
            self.monitor_op = lambda curr, best: curr > best + min_delta

    def step(self, current_score, model=None):
        if self.best_score is None:
            self.best_score = current_score
            if self.save_path and model:
                self._save_model(model)
        elif self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            if self.save_path and model:
                self._save_model(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True

    def _save_model(self, model):
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state_dict, self.save_path)
