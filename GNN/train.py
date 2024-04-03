import os
import time
from typing import Optional

import torch
import torch.nn as nn
os.environ['TORCH'] = torch.__version__
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from tqdm import tqdm
from util import obj


class Trainer:
    def __init__(self,
                 gpu_id: int,
                 hyper_param_dict: dict,
                 model: nn.Module,
                 trainloader: DataLoader,
                 valloader: DataLoader,
                 testloader: DataLoader,
                 num_out_classes: int,
                 model_ckpt_path: str,
                 dovalidate: bool):

        self.gpu_id = gpu_id
        self.h_params = hyper_param_dict
        self.model = model
        self.model_ckptpath = model_ckpt_path
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.criterion = nn.HuberLoss() #nn.MSELoss()  #nn.MSELoss()  # #nn.L1Loss()
        self.doval = dovalidate

        # Optimiser
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.h_params.lr,
            #betas=self.h_params.betas,
        )
        
        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=0.0001,
            min_lr=1e-8,
            verbose=True,
        )
        #self.lr_scheduler = optim.lr_scheduler.StepLR(
        #    self.optimizer, self.const["lr_step_size"]
        #)
        
        # Metrics
        self.train_acc = torchmetrics.R2Score(
            num_outputs=num_out_classes,
            multioutput="uniform_average", #"variance_weighted"
        ).to(self.gpu_id)

        self.valid_acc = torchmetrics.R2Score(
            num_outputs=num_out_classes,
            multioutput="uniform_average", #"variance_weighted" # "raw_values"
        ).to(self.gpu_id)


    def _run_batch(self, src: torch.Tensor, tgt: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        out = self.model(src)
        loss = self.criterion(out, tgt)
        #print(loss)
        loss.backward()
        self.optimizer.step()
        
        self.train_acc.update(out, tgt)
        #print(loss, loss.item())
        return loss.item()

    def _run_epoch(self, epoch: int) -> None:
        batch_losses = []
        train_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        nbatch = len(self.trainloader)
        start = float(time.time())
        for ibatch in tqdm(range(nbatch), ascii=True):
            #print(f"Batch: {ibatch}")
            data_batched = next(iter(self.trainloader))
            src = data_batched.to(self.gpu_id)
            tgt = data_batched.y.to(self.gpu_id)
            #print(src.get_device(), tgt.get_device())
            loss_batch = self._run_batch(src, tgt)
            batch_losses.append(loss_batch)
            #print(loss_batch.get_device())
            train_loss += loss_batch
            
            if self.doval:
                with torch.no_grad():
                    data_batched = next(iter(self.valloader))
                    src = data_batched.to(self.gpu_id)
                    tgt = data_batched.y.to(self.gpu_id)
                    out = self.model(src)
                    loss_batch = self.criterion(out, tgt).item()
                    val_loss += loss_batch
                    self.valid_acc.update(out, tgt)
                
        train_loss = train_loss/nbatch
        train_acc  = self.train_acc.compute().item()
        val_loss   = val_loss / len(self.valloader)
        val_acc    = self.valid_acc.compute().item()
        LRate      = self.optimizer.param_groups[0]['lr']
        self.lr_scheduler.step(train_loss)
        
        end = float(time.time())
        print(
            f"\n[GPU{self.gpu_id}] Epoch {epoch:2d} | Batchsize: {self.h_params.batchlen} | Steps: {len(self.trainloader)} | LR: {LRate:.4f} | Loss: {train_loss:.4f} | R2: {train_acc:.2f} | Val-Loss: {val_loss:.4f} | Val-R2: {val_acc:.2f} | time: {round((end - start)/60,4)} min",
            flush=True,
        )

        self.train_acc.reset()
        if self.doval: 
            self.valid_acc.reset()

        return batch_losses, train_loss, train_acc, val_loss, val_acc, LRate


    def _save_checkpoint(self, epoch: int):
        ckp = self.model.state_dict()
        model_path = os.path.join(self.model_ckptpath, f"GNN_single_epoch{epoch}.pt")
        torch.save(ckp, model_path)

        
    def train(self):
        list_batch_loss = []
        list_train_loss = []
        list_val_loss   = []
        list_train_acc  = []
        list_val_acc    = []
        list_LRate      = []
        self.model.train()
        for epoch in range(self.h_params.nepochs):
            batch_losses, train_loss, train_acc, val_loss, val_acc, LRate = self._run_epoch(epoch)
            list_batch_loss += batch_losses
            list_train_loss.append(train_loss)
            list_train_acc.append(train_acc)
            list_val_loss.append(val_loss)
            list_val_acc.append(val_acc)
            list_LRate.append(LRate)
            if epoch % self.h_params.save_every == 0:
                self._save_checkpoint(epoch)
        # save last epoch
        self._save_checkpoint(self.h_params.nepochs - 1)
        history = {
            "batch": {
                "loss": list_batch_loss,
            },
            "epoch": {
                "loss": {
                    "train": list_train_loss,
                    "val"  : list_val_loss,
                },
                "accuracy": {
                    "train": list_train_acc,
                    "val"  : list_val_acc,
                },
                "LR": list_LRate,
            },
        }
        return history

        
    def test(self, final_model_path: str):
        #self.model = torch.load(final_model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(torch.load(final_model_path))
        self.model.eval()
        preds = None
        trues = None
        start = float(time.time())
        with torch.no_grad():
            for i, batched_data in enumerate(self.testloader):
                src = batched_data.to(self.gpu_id)
                tgt = batched_data.y.to(self.gpu_id)
                out = self.model(src)
                preds = out if i==0 else torch.cat([preds, out])
                trues = tgt if i==0 else torch.cat([trues, tgt])
                
                self.valid_acc.update(out, tgt)
        end = float(time.time())
        print(
            f"[GPU{self.gpu_id}] Test Acc: {100 * self.valid_acc.compute().item():.4f}% ... in {(end-start)/60} mins"
        )
        return trues, preds


from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

class TrainerDDP(Trainer):
    def __init__(self,
                 gpu_id: int,
                 hyper_param_dict: dict,
                 model: nn.Module,
                 trainloader: DataLoader,
                 sampler_train: DistributedSampler,
                 valloader: DataLoader,
                 testloader: DataLoader,
                 num_out_classes: int,
                 model_ckpt_path: str,
                 dovalidate: bool):
        super(TrainerDDP, self).__init__(gpu_id, 
                                         hyper_param_dict, 
                                         model, 
                                         trainloader, 
                                         valloader, 
                                         testloader, 
                                         num_out_classes, 
                                         model_ckpt_path,
                                         dovalidate)

        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.model = self.model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])

        #self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)
        self.sampler_train = sampler_train

    def _save_checkpoint(self, epoch: int):
        ckp = self.model.module.state_dict()
        model_path = os.path.join(self.model_ckptpath, f"GNN_DDP_epoch{epoch}.pt")
        torch.save(ckp, model_path)

    def train(self):
        self.model.train()
        for epoch in range(self.h_params.nepochs):
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            self.sampler_train.set_epoch(epoch)

            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.h_params.save_every == 0:
                self._save_checkpoint(epoch)
        # save last epoch
        self._save_checkpoint(self.h_params.nepochs - 1)



class TrainerDDPtorchrun(TrainerDDP):
    def __init__(self,
                 gpu_id: int,
                 hyper_param_dict: dict,
                 model: nn.Module,
                 trainloader: DataLoader,
                 sampler_train: DistributedSampler,
                 valloader: DataLoader,
                 testloader: DataLoader,
                 num_out_classes: int,
                 model_ckpt_path: str,
                 dovalidate: bool):
        #self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.epochs_run = 0
        super().__init__(gpu_id,
                         hyper_param_dict,
                         model,
                         trainloader,
                         sampler_train,
                         valloader,
                         testloader,
                         num_out_classes,
                         model_ckpt_path,
                         dovalidate)
        def _save_snapshot(self, epoch: int):
            snapshot = dict(
                EPOCHS=epoch,
                MODEL_STATE=self.model.state_dict(),
                OPTIMIZER=self.optimizer.state_dict(),
                LR_SCHEDULER=self.lr_scheduler.state_dict(),
            )
            model_path = f"{self.model_ckptpath}/snapshot.pt"
            torch.save(snapshot, model_path)

        def _load_snapshot(self, path: str):
            snapshot = torch.load(path, map_location="cpu")
            self.epochs_run = snapshot["EPOCHS"] + 1
            self.model.load_state_dict(snapshot["MODEL_STATE"])
            self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
            self.lr_scheduler.load_state_dict(snapshot["LR_SCHEDULER"])
            print(
                f"[GPU{self.gpu_id}] Resuming training from snapshot at Epoch {snapshot['EPOCHS']}"
            )

        def train(self):
            snapshot_path = f"{self.model_ckptpath}/snapshot.pt"
            if Path(snapshot_path).exists():
                print("Loading snapshot")
                self._load_snapshot(snapshot_path)
                
            self.model.train()
            for epoch in range(self.epochs_run, self.h_params.nepochs):
                # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
                self.sampler_train.set_epoch(epoch)

                self._run_epoch(epoch)
                # only save once on master gpu
                if self.gpu_id == 0 and epoch % self.const["save_every"] == 0:
                    self._save_snapshot(epoch)
            # save last epoch
            self._save_checkpoint(self.h_params.nepochs - 1)
