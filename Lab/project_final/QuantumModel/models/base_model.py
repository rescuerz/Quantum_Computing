import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pennylane as qml
import pennylane.numpy as np
import tqdm
import matplotlib.pyplot as plt
import warnings
import os

class BaseModel(torch.nn.Module):
    def __init__(self, **hp):
        super().__init__()
        self.hp = hp

        self.default_use_cuda = torch.cuda.is_available()
        self.cuda_device = torch.device('cuda:0')

    def set_hp(self, **hp):
        self.hp = hp

    def _set_fit_kwargs(self, **kwargs):
        optimizer_parameters = kwargs.get("optimizer_parameters",
                                          {
                                              "lr": 0.01,
                                              "momentum": 0.9,
                                              "nesterov": True,
                                              "weight_decay": 10 ** -6
                                          })
        kwargs["optimizer_parameters"] = optimizer_parameters
        optimizer = kwargs.get("optimizer",
                               optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_parameters)
                               )
        kwargs["optimizer"] = optimizer

        criterion = kwargs.get("criterion", nn.MSELoss())
        kwargs["criterion"] = criterion
        return kwargs

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        kwargs = self._set_fit_kwargs(**kwargs)

        batch_size = kwargs.get("batch_size", 32)

        train_data_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True, drop_last=True
        )

        if X_val is not None and y_val is not None:
            val_data_loader = torch.utils.data.DataLoader(
                list(zip(X_val, y_val)), batch_size=batch_size, shuffle=True, drop_last=True
            )
        else:
            val_data_loader = None

        history = {
            'epochs': [],
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }
        if kwargs.get("use_cuda", self.default_use_cuda):
            self.move_on_gpu()

        epochs_id = range(kwargs.get("epochs", 100))

        verbose = kwargs.get("verbose", False)
        if verbose:
            progress = tqdm.tqdm(
                epochs_id,
                unit="epoch"
            )
            kwargs["progress"] = progress
        else:
            progress = None
        for epoch in epochs_id:
            history["epochs"].append(epoch)

            train_loss, train_acc = self.do_epoch(train_data_loader, **kwargs)
            history["train_loss"].append(train_loss.cpu().detach().numpy())
            history["train_acc"].append(train_acc)

            if val_data_loader is not None:
                val_loss, val_acc = self.do_epoch(val_data_loader, backprop=False, **kwargs)
                history["val_loss"].append(val_loss.cpu().detach().numpy())
                history["val_acc"].append(val_acc)

            if verbose:
                progress.update()
                progress.set_postfix_str(
                    f"epochs: {epoch} train_acc: {train_acc:.2f} val_acc: {val_acc:.2f} "
                    f"train_loss: {train_loss:.2f} val_loss: {val_loss:.2f}"
                )
        if verbose:
            progress.close()
        self.move_on_cpu()
        return history

    def do_epoch(self, data_loader, **kwargs):
        if kwargs.get("scheduler", False):
            kwargs["scheduler"].step()

        epoch_mean_loss = 0
        epoch_mean_acc = 0
        for j, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.float(), targets.float()
            if kwargs.get("use_cuda", self.default_use_cuda):
                [inputs, targets] = self.move_on_gpu(inputs, targets)

            batch_loss = self.do_batch(inputs, targets, **kwargs)
            epoch_mean_loss = (j * epoch_mean_loss + batch_loss) / (j + 1)
            batch_acc = self.score(inputs, targets, kwargs.get("use_cuda", self.default_use_cuda))
            epoch_mean_acc = (j * epoch_mean_acc + batch_acc) / (j + 1)

        return epoch_mean_loss, epoch_mean_acc

    def do_batch(self, inputs, targets, **kwargs):
        # Use model.zero_grad() instead of optimizer.zero_grad()
        # Otherwise, variables that are not optimized won't be cleared
        self.zero_grad()
        output = self(inputs)

        loss = self.apply_criterion(output, targets, **kwargs)

        if kwargs.get("backprop", True):
            loss.backward()
            kwargs["optimizer"].step()

        return loss

    def apply_criterion(self, output, targets, **kwargs):
        return kwargs["criterion"](output, targets)

    def predict(self, X, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.default_use_cuda
        X = torch.tensor(X).float()
        if use_cuda:
            [X] = self.move_on_gpu(X)
        else:
            [X] = self.move_on_cpu(X)
        y_pred = self(X)
        return torch.argmax(y_pred, axis=1).cpu().detach().numpy()

    def score(self, X, y, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.default_use_cuda

        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        if use_cuda:
            [X, y] = self.move_on_gpu(X, y)
        else:
            [X, y] = self.move_on_cpu(X, y)

        return np.mean(self.predict(X, use_cuda) == torch.argmax(y, axis=1).cpu().detach().numpy())

    def move_on_gpu(self, *objs):
        gpu_objs = []
        if torch.cuda.is_available():
            self.to(self.cuda_device)
            for obj in objs:
                gpu_objs.append(obj.to(self.cuda_device))
        else:
            warnings.warn("Cuda is not available on this machine")
        return gpu_objs

    @staticmethod
    def move_on_cpu(*objs):
        cpu_objs = []
        for obj in objs:
            cpu_objs.append(obj.to("cpu"))
        return cpu_objs

    @staticmethod
    def show_history(history: dict, **kwargs):
        epochs = history['epochs']
        
        # 打印训练历史数据, 打印到result.csv文件中, 要求不覆盖原先的内容
        with open('result.csv', 'a') as f:
            f.write("-------------------------------\n")
            f.write(f"{'Epoch':^6} {'Train Acc':^10} {'Val Acc':^10} {'Train Loss':^10} {'Val Loss':^10}\n")
            f.write("-" * 50 + "\n")
            for ep, tr_acc, val_acc, tr_loss, val_loss in zip(
                epochs,
                history['train_acc'],
                history['val_acc'],
                history['train_loss'],
                history['val_loss']
            ):
                f.write(f"{ep:^6d} {tr_acc:^10.4f} {val_acc:^10.4f} {tr_loss:^10.4f} {val_loss:^10.4f}\n")
            f.write("-------------------------------\n")

        print("\n训练历史:")
        print(f"{'Epoch':^6} {'Train Acc':^10} {'Val Acc':^10} {'Train Loss':^10} {'Val Loss':^10}")
        print("-" * 50)
        for ep, tr_acc, val_acc, tr_loss, val_loss in zip(
            epochs,
            history['train_acc'],
            history['val_acc'],
            history['train_loss'],
            history['val_loss']
        ):
            print(f"{ep:^6d} {tr_acc:^10.4f} {val_acc:^10.4f} {tr_loss:^10.4f} {val_loss:^10.4f}")
        print()

        # 原有的绘图代码
        fig, axes = plt.subplots(2, 1)

        axes[0].set_title('Train accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, history['train_acc'], label='Train')
        axes[0].plot(epochs, history['val_acc'], label='Validation')
        axes[0].legend()

        axes[1].set_title('Train loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        train_loss = [loss.detach().numpy() if torch.is_tensor(loss) else loss for loss in history['train_loss']]
        val_loss = [loss.detach().numpy() if torch.is_tensor(loss) else loss for loss in history['val_loss']]
        axes[1].plot(epochs, train_loss, label='Train')
        axes[1].plot(epochs, val_loss, label='Validation')
        plt.tight_layout()
        if kwargs.get("saving", True):
            os.makedirs(f"figures", exist_ok=True)
            plt.savefig(f"figures/training_history_{kwargs.get('name', '')}.png", dpi=300)

        # if kwargs.get("show", False):
        #     plt.show(block=kwargs.get("block", True))

