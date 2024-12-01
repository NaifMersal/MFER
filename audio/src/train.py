import os
import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot, accuracy, device
from torch.utils.tensorboard import SummaryWriter


def to_device(data , device = "cuda" if torch.cuda.is_available()else "cpu"):
    """
    Efficiently moves data to the specified device using type mapping and iteration.
    
    Args:
        data: Input data of any type (tensor, dict, list, tuple, etc.)
        device: Target device ('cuda', 'cpu', or torch.device object)
    
    Returns:
        Data moved to the specified device while preserving its structure
    """
    # Type handlers map
    type_handlers = {
        torch.Tensor: lambda x: x.to(device),
        dict: lambda x: {k: to_device(v, device) for k, v in x.items()},
        list: lambda x: [to_device(item, device) for item in x],
        tuple: lambda x: tuple(to_device(item, device) for item in x),
    }
    

    
    # Handle objects with custom .to() method
    if hasattr(data, 'to'):
        return data.to(device)
        # Fast path for common cases
    # error if not exist
    data_type = type(data)
    return type_handlers[data_type](data)
    # Return unchanged for unsupported types


def train_one_epoch(train_dataloader, model, optimizer, loss, scaler, accumulation_steps, topk=(1,)):
    model.train()
    train_loss = 0.0
    accs = [0] * len(topk)
    
    for batch_idx, (data, target) in tqdm(
            enumerate(train_dataloader),
            desc="Training",
            total=len(train_dataloader),
            leave=True,
            ncols=80):
        
        data, target = to_device(data, device), to_device(target, device)
        if scaler:
            with torch.autocast(device_type=device, dtype=torch.float16):
                output = model(data)
                loss_value = loss(output, target)
            scaler.scale(loss_value).backward()

            if (batch_idx+1) % accumulation_steps == 0 or (batch_idx+1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            output = model(data)
            loss_value = loss(output, target)
            loss_value.backward()
            
            if (batch_idx+1) % accumulation_steps == 0 or (batch_idx+1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss))
        batch_accs = accuracy(output, target, topk=topk)
        for i, acc in enumerate(batch_accs):
            accs[i] += acc.item()  # Convert tensor to Python number
    
    return train_loss, [acc/len(train_dataloader) for acc in accs]

def valid_one_epoch(valid_dataloader, model, loss, topk=(1,)):
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        accs = [0] * len(topk)

        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80):
            data, target = to_device(data, device), to_device(target, device)
            output = model(data)
            loss_value = loss(output, target)
            
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss))
            batch_accs = accuracy(output, target, topk=topk)
            for i, acc in enumerate(batch_accs):
                accs[i] += acc.item()  # Convert tensor to Python number

    return valid_loss, [acc/len(valid_dataloader) for acc in accs]

def optimize(data_loaders, model, optimizer, loss, s_epoch, n_epochs, model_name, step, topk=(1,), accumulation_steps=1, use_amp=True, interactive_tracking=False,checkpoints_dir='checkpoints',run_logs=False):
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_acc_max = None
    logs = {}
    scaler = torch.amp.GradScaler() if use_amp else None
    model = model.to(device)

    if run_logs:
        path = f"runs/{model_name}"
        writer = SummaryWriter(path)
        if not os.path.isfile(path):
            writer.add_text('model', str(model))
            writer.add_text('optimizer', str(optimizer))

    for epoch in range(s_epoch, s_epoch + n_epochs):
        train_loss, train_accs = train_one_epoch(data_loaders["train"], model, optimizer, loss, scaler, accumulation_steps, topk)
        valid_loss, valid_accs = valid_one_epoch(data_loaders["valid"], model, loss, topk)

        if interactive_tracking:
            # Ensure all values are Python numbers, not tensors
            logs["loss"] = float(train_loss)
            logs["val_loss"] = float(valid_loss)
            logs["acc"] = float(train_accs[0])
            logs["val_acc"] = float(valid_accs[0])
            logs["lr"] = float(optimizer.param_groups[0]["lr"])
            liveloss.update(logs)
            liveloss.send()

        if run_logs:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/valid", valid_loss, epoch)
            for i, k in enumerate(topk):
                writer.add_scalar(f"Acc{k}/train", train_accs[i], epoch)
                writer.add_scalar(f"Acc{k}/valid", valid_accs[i], epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Dynamic accuracy printing
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}", end='')
        for i, k in enumerate(topk):
            print(f"\tTraining Acc@{k}: {train_accs[i]:.3f} \tValidation Acc@{k}: {valid_accs[i]:.3f}", end='')
        print()

        if valid_acc_max is None or valid_acc_max <= valid_accs[0]:
            print(f"New max accuracy: {valid_accs[0]:.6f}. Saving model ...")
            torch.save({
                'epochs': epoch,
                'model_state_dict': model.state_dict(),
            }, f'{checkpoints_dir}/best_{model_name}.pt')
            valid_acc_max = valid_accs[0]

        torch.save({
            'epochs': epoch,
            'model_state_dict': model.state_dict(),
        }, f'{checkpoints_dir}/last_{model_name}.pt')
        step(valid_loss, epoch)

    if run_logs:
        writer.flush()
        writer.close()



