import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from utils_spot import cosine_scheduler


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, ce_weight_schedule,log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 400
    accum_iter = args.accum_iter
    loss_mage=0
    loss_mage_spot=0
    optimizer.zero_grad()
    train_epoch_size = len(data_loader)





    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch_data,mask_crf) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = batch_data.to(device, non_blocking=True)  # Use only the first tensor (images)
        mask_crf = mask_crf.to(device, non_blocking=True)
        # LR Scheduler Adjustment (if necessary)
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        global_step = epoch * train_epoch_size + data_iter_step
        ce_weight = ce_weight_schedule[global_step]

        with torch.cuda.amp.autocast():
            if args.use_decs:
                # breakpoint()
                loss_comb, _, _,_,_,_ = model(samples,mask_crf)

                loss_mage, loss_mage_spot,ce_loss = loss_comb
                
                loss = loss_mage + ce_weight*ce_loss + 0.3*loss_mage_spot

                
                # breakpoint()
                # loss=loss_mage+(0.5*loss_mage_spot)
                # loss=loss_mage
            else:
                loss, _, _,_,_,_ = model(samples)

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
            log_writer.add_scalar('ce', ce_weight, epoch_1000x)
            if args.use_decs:
                log_writer.add_scalar('ce loss', ce_loss, epoch_1000x)
                log_writer.add_scalar('loss_mage', loss_mage, epoch_1000x)
                log_writer.add_scalar('loss_spot', loss_mage_spot, epoch_1000x)
                

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
