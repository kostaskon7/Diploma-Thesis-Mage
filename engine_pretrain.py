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
#     # for batch in data_loader:
#     #     print(f"Number of items in batch: {len(batch)}")
#     #     print([type(item) for item in batch])
#     #     break  # Only inspect the first batch
#     # for batch in data_loader:
#     #     print(f"Number of items in batch: {len(batch)}")
#     #     print([type(item) for item in batch])
#     #     break  # Inspect just the first batch

#     # for batch in data_loader:
#     #     for i, item in enumerate(batch):
#     #         if isinstance(item, torch.Tensor):
#     #             print(f"Item {i}: shape {item.shape}")
#     #         else:
#     #             print(f"Item {i}: type {type(item)}")
#     #             if(i==1):
#     #                 print(item)
#     #     break  # Inspect just the first batch





#     for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         print(samples.shape)
#     # for data_iter_step, (samples, mask_instance, mask_class, mask_ignore) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

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

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = batch_data[0].to(device, non_blocking=True)  # Use only the first tensor (images)
        print(samples.shape)
        # LR Scheduler Adjustment (if necessary)
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
