import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


# def train_one_epoch(model: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler,
#                     log_writer=None,
#                     args=None):
#     model.train(True)
#     metric_logger = misc.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 20

#     accum_iter = args.accum_iter

#     optimizer.zero_grad()

#     if log_writer is not None:
#         print('log_dir: {}'.format(log_writer.log_dir))

#     # Add this code before your training loop to inspect the data_loader output
#     for batch in data_loader:
#         print(f"Number of items in batch: {len(batch)}")
#         print([type(item) for item in batch])
#         break  # Only inspect the first batch


#     # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#     for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         samples = batch_data[0].to(device, non_blocking=True)  # Use only the first tensor

#         # we use a per iteration (instead of per epoch) lr scheduler
#         if data_iter_step % accum_iter == 0:
#             lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

#         samples = samples.to(device, non_blocking=True)

#         with torch.cuda.amp.autocast():
#             loss, _, _ = model(samples)

#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)

#         loss /= accum_iter
#         loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(),
#                     update_grad=(data_iter_step + 1) % accum_iter == 0)
#         if (data_iter_step + 1) % accum_iter == 0:
#             optimizer.zero_grad()

#         torch.cuda.synchronize()

#         metric_logger.update(loss=loss_value)

#         lr = optimizer.param_groups[0]["lr"]
#         metric_logger.update(lr=lr)

#         loss_value_reduce = misc.all_reduce_mean(loss_value)
#         if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
#             """ We use epoch_1000x as the x-axis in tensorboard.
#             This calibrates different curves when batch size changes.
#             """
#             epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
#             log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
#             log_writer.add_scalar('lr', lr, epoch_1000x)


#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import torch
from torch.nn.utils import clip_grad_norm_

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    train_epoch_size = len(data_loader)
    log_interval = train_epoch_size // 5  # or adjust as per your requirement

    for batch, batch_data in enumerate(data_loader):
        # Assuming the first item in batch_data is the tensor of samples
        samples = batch_data[0].to(device, non_blocking=True)
        global_step = epoch * train_epoch_size + batch

        # Adjust learning rate if necessary
        # (Add your learning rate adjustment logic here if required)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples)

        loss_scaler(loss, optimizer, parameters=model.parameters())

        if batch % log_interval == 0:
            loss_value = loss.item()
            print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:.6f}'.format(
                epoch + 1, batch, train_epoch_size, loss_value))

            if log_writer is not None:
                log_writer.add_scalar('train_loss', loss_value, global_step)

    return {"loss": loss_value}
