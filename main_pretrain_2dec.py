import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mage_2dec

from engine_pretrain import train_one_epoch
from spot.datasets import COCO2017

from spot.ocl_metrics import UnsupervisedMaskIoUMetric, ARIMetric
from spot.utils_spot import inv_normalize, cosine_scheduler, visualize, bool_flag, load_pretrained_encoder
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.utils as vutils
import math
import torchvision.utils as vutils
import sys
import cv2
# import torch_xla.core.xla_model as xm





def get_args_parser():
    parser = argparse.ArgumentParser('MAGE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mage_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')

    parser.add_argument('--vqgan_ckpt_path', default='vqgan_jax_strongaug.ckpt', type=str,
                        help='Path to the VQGAN checkpoint file')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # MAGE params
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--mask_ratio_max', type=float, default=1.0,
                        help='Maximum mask ratio')
    parser.add_argument('--mask_ratio_mu', type=float, default=0.55,
                        help='Mask ratio distribution peak')
    parser.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Mask ratio distribution std')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--use_decs', default=None,type=int,
                    help='2 decoders used')
    

    # Spot
    parser.add_argument('--num_dec_blocks', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--num_slots', type=int, default=7)
    parser.add_argument('--slot_size', type=int, default=256)
    parser.add_argument('--mlp_hidden_size', type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--pos_channels', type=int, default=4)
    parser.add_argument('--num_cross_heads', type=int, default=None)

    parser.add_argument('--dec_type',  type=str, default='transformer', help='type of decoder transformer or mlp')
    parser.add_argument('--cappa', type=float, default=-1)
    parser.add_argument('--mlp_dec_hidden',  type=int, default=2048, help='Dimension of decoder mlp hidden layers')
    parser.add_argument('--use_slot_proj',  type=bool_flag, default=True, help='Use an extra projection before MLP decoder')
    
    parser.add_argument('--train_permutations',  type=str, default='random', help='which permutation')
    parser.add_argument('--eval_permutations',  type=str, default='standard', help='which permutation')

    parser.add_argument('--truncate',  type=str, default='none', help='bi-level or fixed-point or none')
    parser.add_argument('--init_method', default='shared_gaussian', help='embedding or shared_gaussian')


    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    # if device == 'tpu':
    #     # First, make sure you've installed PyTorch XLA
    #     device = xm.xla_device()

    # else:
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])


    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # # #print(dataset_train)

    # if True:  # args.distributed:
    #     num_tasks = misc.get_world_size()
    #     global_rank = misc.get_rank()
    #     sampler_train = torch.utils.data.DistributedSampler(
    #         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )
    #     print("Sampler_train = %s" % str(sampler_train))
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # if global_rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    # else:
    #     log_writer = None

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    log_writer = SummaryWriter(log_dir=args.log_dir)
    train_sampler = None
    val_sampler = None


    train_dataset = COCO2017(root=args.data_path, split='train', image_size=256, mask_size=256)

    
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, shuffle=True, drop_last=True, batch_size=args.batch_size, pin_memory=True,num_workers= 4)#,collate_fn=custom_collate_fn)

    val_dataset = COCO2017(root=args.data_path, split='val', image_size=256, mask_size=256)
    val_loader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, shuffle=False, drop_last=False, batch_size=args.batch_size, pin_memory=True,num_workers= 4)#,collate_fn=custom_collate_fn)

    # define the model
    vqgan_ckpt_path = args.vqgan_ckpt_path

    model = models_mage_2dec.__dict__[args.model](mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
                                             mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
                                             vqgan_ckpt_path=vqgan_ckpt_path,args=args)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # if os.path.isfile(args.resume):
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     start_epoch = checkpoint['epoch']
    #     best_val_loss = checkpoint['best_val_loss']
    #     best_val_ari = checkpoint['best_val_dec_ari']
    #     best_val_ari_slot = checkpoint['best_val_default_ari']
    #     best_mbo_c = checkpoint['best_mbo_c']
    #     best_mbo_i = checkpoint['best_mbo_i']
    #     best_fg_iou = checkpoint['best_fg_iou']
    #     best_mbo_c_slot = checkpoint['best_mbo_c_slot']
    #     best_mbo_i_slot = checkpoint['best_mbo_i_slot']
    #     best_fg_iou_slot = checkpoint['best_fg_iou_slot']
    #     best_epoch = checkpoint['best_epoch']
    #     model.load_state_dict(checkpoint['model'], strict=True)
    #     msg = model.load_state_dict(checkpoint['model'], strict=True)
    #     print(msg)
    # else:
    print('No checkpoint_path found')
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0
    best_val_ari = 0
    best_val_ari_slot = 0
    best_mbo_c = 0
    best_mbo_i = 0
    best_fg_iou= 0 
    best_mbo_c_slot = 0
    best_mbo_i_slot = 0
    best_fg_iou_slot= 0 



    MBO_c_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
    MBO_i_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
    fg_iou_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).to(device)
    ari_metric = ARIMetric(foreground = True, ignore_overlaps = True).to(device)
    
    MBO_c_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
    MBO_i_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
    fg_iou_slot_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).to(device)
    ari_slot_metric = ARIMetric(foreground = True, ignore_overlaps = True).to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, 2):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        # if args.output_dir and (epoch % 40 == 0 or epoch + 1 == args.epochs):
        #     misc.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch)

        # misc.save_model_last(
        #     args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #     loss_scaler=loss_scaler, epoch=epoch)
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                 'epoch': epoch,}

        # if args.output_dir and misc.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")



        visualize_per_epoch = 1#int(args.epochs*args.eval_viz_percent)
        val_epoch_size = len(val_loader)

        with torch.no_grad():
            model.eval()

            counter = 0
    
            for batch, (image, true_mask_i, true_mask_c, mask_ignore) in enumerate(tqdm(val_loader)):
                image = image.cuda()
                true_mask_i = true_mask_i.cuda()
                true_mask_c = true_mask_c.cuda()
                mask_ignore = mask_ignore.cuda() 
                
                batch_size = image.shape[0]
                counter += batch_size
    
                val_loss,_,_,default_slots_attns, dec_slots_attns,logits = model(image)
                
                val_loss_mage, val_loss_spot = val_loss
                codebook_emb_dim=256
                logits = logits[:, 8:, :model.codebook_size]
                # logits = logits[:, 1:, :model.codebook_size]

                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                reconstructed_indices = torch.argmax(probabilities, dim=-1)
                z_q = model.vqgan.quantize.get_codebook_entry(reconstructed_indices, shape=(batch_size, 16, 16, codebook_emb_dim))
                gen_images = model.vqgan.decode(z_q)


                gen_img_list = []
                gen_images_batch = gen_images.detach().cpu()
                gen_img_list.append(gen_images_batch)
                orig_images_batch=image.detach().cpu()

                # Save images
                for b_id in range(batch_size):
                    # Apply inverse normalization
                    # inv_gen_img = inv_normalize(gen_images_batch[b_id])
                    inv_gen_img=gen_images_batch[b_id]
                    # inv_orig_img = inv_normalize(orig_images_batch[b_id])
                    inv_orig_img = orig_images_batch[b_id]

                    # Convert to numpy and save - Generated Image
                    gen_img_np = np.clip(inv_gen_img.numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                    gen_img_np = cv2.cvtColor(gen_img_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(args.output_dir, '{}.png'.format(str(epoch * batch_size + b_id).zfill(5))), gen_img_np)

                    # Convert to numpy and save - Original Image
                    orig_img_np = np.clip(inv_orig_img.numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                    orig_img_np = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(args.output_dir, 'orig_{}.png'.format(str(epoch * batch_size + b_id).zfill(5))), orig_img_np)
                                    ################ Recon

                default_slots_attns = default_slots_attns.transpose(-1, -2).reshape(batch_size, 7, 16, 16)
                dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(batch_size, 7, 16, 16)
                # default_slots_attns=default_slots_attns.unsqueeze(3)
                # dec_slots_attns=dec_slots_attns.unsqueeze(3)


                default_attns = F.interpolate(default_slots_attns, size=256, mode='bilinear')
                dec_attns = F.interpolate(dec_slots_attns, size=256, mode='bilinear')
                # dec_attns shape [B, num_slots, H, W]
                default_attns = default_attns.unsqueeze(2)
                dec_attns = dec_attns.unsqueeze(2) # shape [B, num_slots, 1, H, W]
    
                pred_default_mask = default_attns.argmax(1).squeeze(1)
                pred_dec_mask = dec_attns.argmax(1).squeeze(1)
                # print("Unsqueeze")
                # print(default_attns.shape)
                # print(true_mask_i.shape)
                # print(true_mask_c.shape)
                # print("Squeeze")
                # print(pred_default_mask.shape)
                # print(true_mask_i.shape)
                # print(true_mask_c.shape)
    

                # Compute ARI, MBO_i and MBO_c, fg_IoU scores for both slot attention and decoder
                true_mask_i_reshaped = torch.nn.functional.one_hot(true_mask_i).to(torch.float32).permute(0,3,1,2).cuda()
                true_mask_c_reshaped = torch.nn.functional.one_hot(true_mask_c).to(torch.float32).permute(0,3,1,2).cuda()
                pred_dec_mask_reshaped = torch.nn.functional.one_hot(pred_dec_mask).to(torch.float32).permute(0,3,1,2).cuda()
                pred_default_mask_reshaped = torch.nn.functional.one_hot(pred_default_mask).to(torch.float32).permute(0,3,1,2).cuda()
                
                MBO_i_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                MBO_c_metric.update(pred_dec_mask_reshaped, true_mask_c_reshaped, mask_ignore)
                fg_iou_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                ari_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
            
                MBO_i_slot_metric.update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                MBO_c_slot_metric.update(pred_default_mask_reshaped, true_mask_c_reshaped, mask_ignore)
                fg_iou_slot_metric.update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                ari_slot_metric.update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)
    
            ari = 100 * ari_metric.compute()
            ari_slot = 100 * ari_slot_metric.compute()
            mbo_c = 100 * MBO_c_metric.compute()
            mbo_i = 100 * MBO_i_metric.compute()
            fg_iou = 100 * fg_iou_metric.compute()
            mbo_c_slot = 100 * MBO_c_slot_metric.compute()
            mbo_i_slot = 100 * MBO_i_slot_metric.compute()
            fg_iou_slot = 100 * fg_iou_slot_metric.compute()
            
            log_writer.add_scalar('VAL/val_spot', val_loss_mage, epoch)
            log_writer.add_scalar('VAL/val_mage', val_loss_mage, epoch)
            log_writer.add_scalar('VAL/ari (slots)', ari_slot, epoch)
            log_writer.add_scalar('VAL/ari (decoder)', ari, epoch)
            log_writer.add_scalar('VAL/mbo_c', mbo_c, epoch)
            log_writer.add_scalar('VAL/mbo_i', mbo_i, epoch)
            log_writer.add_scalar('VAL/fg_iou', fg_iou, epoch)
            log_writer.add_scalar('VAL/mbo_c (slots)', mbo_c_slot, epoch)
            log_writer.add_scalar('VAL/mbo_i (slots)', mbo_i_slot, epoch)
            log_writer.add_scalar('VAL/fg_iou (slots)', fg_iou_slot, epoch)
            
            #print(args.log_path)
            print('====> Epoch: {:3} \t Loss = {:F}  \t ARI = {:F} \t ARI_slots = {:F} \t mBO_c = {:F} \t mBO_i = {:F} \t fg_IoU = {:F} \t mBO_c_slots = {:F} \t mBO_i_slots = {:F} \t fg_IoU_slots = {:F}'.format(
                epoch, val_loss_mage, ari, ari_slot, mbo_c, mbo_i, fg_iou, mbo_c_slot, mbo_i_slot, fg_iou_slot))
            
            ari_metric.reset()
            MBO_c_metric.reset()
            MBO_i_metric.reset()
            fg_iou_metric.reset()
            MBO_c_slot_metric.reset()
            MBO_i_slot_metric.reset()
            ari_slot_metric.reset()
            fg_iou_slot_metric.reset()
            
            if (val_loss_mage < best_val_loss):
                best_val_loss = val_loss_mage
                
                best_val_ari_slot = ari_slot
                best_mbo_c = mbo_c
                best_mbo_i = mbo_i
                best_fg_iou = fg_iou
                best_mbo_c_slot = mbo_c_slot
                best_mbo_i_slot = mbo_i_slot
                best_fg_iou_slot = fg_iou_slot
                best_epoch = epoch

            if(best_val_ari > ari):
                best_val_ari = ari

            if(best_mbo_c > mbo_c):
                best_mbo_c = mbo_c

                #torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
                
            if epoch%visualize_per_epoch==0 or epoch==args.epochs-1:
                image = inv_normalize(image)
                image = F.interpolate(image, size=256, mode='bilinear')#EDWWWWWWWW HTAN args.mask_size
                rgb_default_attns = image.unsqueeze(1) * default_attns + 1. - default_attns
                rgb_dec_attns = image.unsqueeze(1) * dec_attns + 1. - dec_attns
    
                vis_recon = visualize(image, true_mask_c, pred_dec_mask, rgb_dec_attns, pred_default_mask, rgb_default_attns, N=32)
                grid = vutils.make_grid(vis_recon, nrow=2*7 + 4, pad_value=0.2)[:, 2:-2, 2:-2]#anti gia 7 num_slots
                grid = F.interpolate(grid.unsqueeze(1), scale_factor=0.15, mode='bilinear').squeeze() # Lower resolution
                log_writer.add_image('VAL_recon/epoch={:03}'.format(epoch), grid)
    
            log_writer.add_scalar('VAL/best_loss', best_val_loss, epoch)
    
            checkpoint = {
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_ari': best_val_ari,
                'best_val_ari_slot': best_val_ari_slot,
                'best_mbo_c':best_mbo_c,
                'best_mbo_i':best_mbo_i,
                'best_fg_iou':best_fg_iou,
                'best_mbo_c_slot':best_mbo_c_slot,
                'best_mbo_i_slot':best_mbo_i_slot,
                'best_fg_iou_slot':best_fg_iou_slot,
                'best_epoch': best_epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint-%s.pth" % epoch))

            print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))
        #     if(epoch>25):
        #         break
        # if(epoch>25):
        #     break
    log_writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
