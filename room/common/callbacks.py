import torch


class TrainCallback:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def on_train_begin(self, logs):
        self.model.train()

    def on_train_batch_begin(self, batch, logs):
        pass

    def on_train_batch_end(self, batch, logs):
        pass

    def on_train_end(self, logs):
        pass

    def __call__(self, batch):
        self.on_train_batch_begin(batch, {})
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.on_train_batch_end(batch, {"loss": loss.item()})
        return loss.item()
