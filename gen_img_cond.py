import torch
import os
import math
import argparse
import models_mage_2dec
import models_mage
import numpy as np
from tqdm import tqdm
import cv2
from spot.datasets import COCO2017
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import torch
import torch.nn.functional as F
from spot.utils_spot import inv_normalize, cosine_scheduler, visualize, bool_flag, load_pretrained_encoder
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
# from kmeans_pytorch import kmeans, kmeans_predict
from joblib import load
import torch








def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(temperature * np.random.gumbel(size=probs.shape)).cuda()
    sorted_confidence, _ = torch.sort(confidence, axis=-1)
    # Obtains cut off threshold given the mask lengths.
    cut_off = sorted_confidence[:, mask_len.long()-1:mask_len.long()]
    # Masks tokens with lower confidence.
    masking = (confidence <= cut_off)
    return masking


def gen_image(model, image, bsz, seed, num_iter=12, choice_temperature=4.5,per_iter=False,with_mask_vis=False,data_used='coco',slot_vis=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    codebook_emb_dim = 256
    codebook_size = 1024
    mask_token_id = model.mask_token_label
    unknown_number_in_the_beginning = 256
    _CONFIDENCE_OF_KNOWN_TOKENS = +np.inf

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    image=image.cuda()

    # Assuming you've saved the cluster centers as 'cluster_centers.pth'
    # cluster_centers = torch.load('cluster_centers.pth')
    # # # Ensure the cluster centers are on the correct device
    # cluster_centers = cluster_centers.cuda()



    # Load the model
    kmeans_model = load(args.kmeans_path)

 



############################ Create Slot vis
    if slot_vis:
        if data_used == 'coco' :
    
            val_loss,_,_,default_slots_attns, dec_slots_attns,logits = model(image)

            default_slots_attns = default_slots_attns.transpose(-1, -2).reshape(args.batch_size, 7, 16, 16)
            dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(args.batch_size, 7, 16, 16)
            # default_slots_attns=default_slots_attns.unsqueeze(3)
            # dec_slots_attns=dec_slots_attns.unsqueeze(3)


            default_attns = F.interpolate(default_slots_attns, size=256, mode='bilinear')
            dec_attns = F.interpolate(dec_slots_attns, size=256, mode='bilinear')
            # dec_attns shape [B, num_slots, H, W]
            default_attns = default_attns.unsqueeze(2)
            dec_attns = dec_attns.unsqueeze(2) # shape [B, num_slots, 1, H, W]

            pred_default_mask = default_attns.argmax(1).squeeze(1)
            pred_dec_mask = dec_attns.argmax(1).squeeze(1)

            image_int = F.interpolate(image, size=256, mode='bilinear')#EDWWWWWWWW HTAN args.mask_size
            rgb_default_attns = image_int.unsqueeze(1) * default_attns + 1. - default_attns
            rgb_dec_attns = image_int.unsqueeze(1) * dec_attns + 1. - dec_attns

            vis_recon = visualize(image_int, true_mask_c, pred_dec_mask, rgb_dec_attns, pred_default_mask, rgb_default_attns, N=32)
            grid = vutils.make_grid(vis_recon, nrow=2*7 + 4, pad_value=0.2)[:, 2:-2, 2:-2]#anti gia 7 num_slots
            grid = F.interpolate(grid.unsqueeze(1), scale_factor=0.15, mode='bilinear').squeeze() # Lower resolution
            log_writer.add_image('VAL_recon/epoch={:03}'.format(1), grid)
        else:
            val_loss,_,_,default_slots_attns, dec_slots_attns,logits = model(image)

            default_slots_attns = default_slots_attns.transpose(-1, -2).reshape(args.batch_size, 7, 16, 16)
            dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(args.batch_size, 7, 16, 16)
            # default_slots_attns=default_slots_attns.unsqueeze(3)
            # dec_slots_attns=dec_slots_attns.unsqueeze(3)


            default_attns = F.interpolate(default_slots_attns, size=256, mode='bilinear')
            dec_attns = F.interpolate(dec_slots_attns, size=256, mode='bilinear')
            # dec_attns shape [B, num_slots, H, W]
            default_attns = default_attns.unsqueeze(2)
            dec_attns = dec_attns.unsqueeze(2) # shape [B, num_slots, 1, H, W]

            pred_default_mask = default_attns.argmax(1).squeeze(1)
            pred_dec_mask = dec_attns.argmax(1).squeeze(1)

            image_int = F.interpolate(image, size=256, mode='bilinear')#EDWWWWWWWW HTAN args.mask_size
            rgb_default_attns = image_int.unsqueeze(1) * default_attns + 1. - default_attns
            rgb_dec_attns = image_int.unsqueeze(1) * dec_attns + 1. - dec_attns

            vis_recon = visualize(image_int, pred_dec_mask, pred_dec_mask, rgb_dec_attns, pred_default_mask, rgb_default_attns, N=32)
            grid = vutils.make_grid(vis_recon, nrow=2*7 + 4, pad_value=0.2)[:, 2:-2, 2:-2]#anti gia 7 num_slots
            grid = F.interpolate(grid.unsqueeze(1), scale_factor=0.15, mode='bilinear').squeeze() # Lower resolution
            log_writer.add_image('VAL_recon/epoch={:03}'.format(1), grid)

    

#########################

    # latent, gt_indices, _, _ = model.forward_encoder(image)
    # latent = model.forward_encoder(image)

    # #slots, attn, init_slots, attn_logits = self.slot_attention(latent[:,1:,:])
    # slots, attn, init_slots, attn_logits = model.slot_attention(latent[:,1:,:])
    
    

    # slots = torch.matmul(attn.transpose(-1, -2), latent[:,1:,:])

    latent,_,_,_=model.forward_encoder_copy(image)
    latent=latent[:,1:,:]

    slots, attn, init_slots, attn_logits = model.slot_attention(latent)
    # slots_pool = torch.matmul(attn.transpose(-1, -2), x)

    attn=attn.clone().detach()
    attn_onehot = torch.nn.functional.one_hot(attn.argmax(2), num_classes=model.slot_attention.num_slots).to(latent.dtype)
    # To add normalization
    # attn_onehot = attn_onehot / torch.sum(attn_onehot+self.epsilon, dim=-2, keepdim=True)
    slots = torch.matmul(attn_onehot.transpose(-1, -2), latent)


    slots = model.slot_proj2(slots)

    # # Assuming 'your_slots_tensor' is your slots tensor with shape [images, num_slots, 256]
    slots_tensor = slots  # Replace with your actual tensor
    slots_2d = slots_tensor.reshape(-1, 768).cpu().numpy()  # Reshape to 2D for prediction

    # Predict cluster assignments
    cluster_assignments = kmeans_model.predict(slots_2d)

    # Replace slots with cluster centers
    centers = kmeans_model.cluster_centers_[cluster_assignments]  # Shape: [images*num_slots, 256]

    breakpoint()

    # Reshape back to the original slots shape
    slots = centers.reshape(-1, slots_tensor.shape[1], 768)  # Use the original num_slots
    # slots = centers.reshape(-1, 256, 7)  # Use the original num_slots

    slots = torch.tensor(slots).cuda()

    # # Find the indices of the maximum values (most important features) from the soft attention
    # max_indices = torch.argmax(attn, dim=-1)

    # # Convert these indices to a one-hot encoding format
    # attn_hard = torch.nn.functional.one_hot(max_indices, num_classes=slots.shape[1]).to(latent.dtype)

    # # Use the one-hot encoded indices to select features
    # slots = torch.matmul(attn_hard.float().transpose(-1, -2), latent)

    # slots = model.slot_proj2(slots)

    # attn=slots.reshape(bsz,slots_tensor.shape[1],256)
    # attn=slots

    # attn_onehot = torch.nn.functional.one_hot(attn.argmax(2), num_classes=model.slot_attention.num_slots).to(latent.dtype)
    # # To add normalization
    # # attn_onehot = attn_onehot / torch.sum(attn_onehot+self.epsilon, dim=-2, keepdim=True)
    # slots = torch.matmul(attn_onehot.transpose(-1, -2), latent)


    # slots = model.slot_proj2(slots)

    # slots=model.slot_proj2(slots)

    initial_token_indices = mask_token_id * torch.ones(bsz, unknown_number_in_the_beginning)

    token_indices = initial_token_indices.cuda()

    for step in range(num_iter):
        cur_ids = token_indices.clone().long()

        token_indices = torch.cat(
            [torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = model.fake_class_label
        token_indices = token_indices.long()
        token_all_mask = token_indices == mask_token_id

        token_drop_mask = torch.zeros_like(token_indices)

        # token embedding
        input_embeddings = model.token_emb(token_indices)

        # encoder
        x = input_embeddings
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)

        

        # if step==0:
        #     print('First iteration')
        #     slots, attn, init_slots, attn_logits = model.slot_attention(x)
        #     # slots_pool = torch.matmul(attn.transpose(-1, -2), x)
        #     slots = model.slot_proj2(slots)

        #     # Assuming 'your_slots_tensor' is your slots tensor with shape [images, num_slots, 256]
        #     slots_tensor = slots  # Replace with your actual tensor
        #     slots_2d = slots_tensor.reshape(-1, 768).cpu().numpy()  # Reshape to 2D for prediction

        #     # Predict cluster assignments
        #     cluster_assignments = kmeans.predict(slots_2d)

        #     # Replace slots with cluster centers
        #     centers = kmeans.cluster_centers_[cluster_assignments]  # Shape: [images*num_slots, 256]

        #     # Reshape back to the original slots shape
        #     slots = centers.reshape(-1, slots_tensor.shape[1], 768)  # Use the original num_slots
        #     slots = torch.tensor(slots).cuda()


        
        # Find top k slots
        # if step==0:
        #     n_top_slots = 6
        #     slots_summed_values = slots.sum(dim=2)
        #     _, top_slot_indices = slots_summed_values.topk(n_top_slots, dim=1)
        #     slots = torch.gather(slots, 1, top_slot_indices.unsqueeze(-1).expand(-1, -1, slots.size(2)))
        #     # slots=slots[:,2,:].unsqueeze(1)

        # if step==0:
        #     slots_reshaped = slots.reshape(-1, 256)

        #     # Predict cluster IDs for each slot
        #     cluster_ids = kmeans_predict(
        #         slots_reshaped, cluster_centers, 'euclidean', device=device
        #     )

        #     # Replace each slot with its corresponding cluster center
        #     # This will use the predicted cluster IDs to gather the appropriate cluster centers
        #     slots_replaced = cluster_centers[cluster_ids]

        #     # Reshape back to the original slots tensor shape
        #     slots_replaced = slots_replaced.view_as(slots)
        #     slots = torch.matmul(attn.transpose(-1, -2), latent[:,1:,:])

        # logits,_ = model.forward_decoder(x, slots_replaced, token_drop_mask, token_all_mask)

        logits,_ = model.forward_decoder(x, slots, token_drop_mask, token_all_mask)
        # logits = logits[:, model.slot_attention.num_slots+1:, :codebook_size]



        # decoder
        # logits,_ = model.forward_decoder(x, slots, token_drop_mask, token_all_mask)
        logits = logits[:, model.slot_attention.num_slots+1:, :codebook_size]
        # logits = logits[:, n_top_slots+1:, :codebook_size]

        # get token prediction
        sample_dist = torch.distributions.categorical.Categorical(logits=logits)
        sampled_ids = sample_dist.sample()

        # get ids for next step
        unknown_map = (cur_ids == mask_token_id)
        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter

        mask_ratio = np.cos(math.pi / 2. * ratio)
        print(mask_ratio)

        # sample ids according to prediction confidence
        probs = torch.nn.functional.softmax(logits, dim=-1)
        selected_probs = torch.squeeze(
            torch.gather(probs, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)

        selected_probs = torch.where(unknown_map, selected_probs.double(), _CONFIDENCE_OF_KNOWN_TOKENS).float()

        mask_len = torch.Tensor([np.floor(unknown_number_in_the_beginning * mask_ratio)]).cuda()
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                 torch.minimum(torch.sum(unknown_map, dim=-1, keepdims=True) - 1, mask_len))

        # Sample masking tokens for next iteration
        masking = mask_by_random_topk(mask_len[0], selected_probs, choice_temperature * (1 - ratio))
        # Masks tokens with lower confidence.
        token_indices = torch.where(masking, mask_token_id, sampled_ids)

  

        if(with_mask_vis):
            batch_size=args.batch_size

            # Assuming token_indices is of shape [32, 256] with 2024 indicating masks
            batch_size, hw = token_indices.shape  # hw is 256 in your case

            # Generate a boolean mask for where token_indices equals 2024
            mask = token_indices == 2024


            # Reshape mask to [batch_size, 1, 16, 16] for interpolation
            mask = mask.view(batch_size, 1, 16, 16)

            # Interpolate mask to [batch_size, 1, 256, 256]
            mask_upsampled = F.interpolate(mask.float(), size=(256, 256), mode='nearest').bool()

            # Expand mask to match image channels (assuming RGB, so repeat 3 times across dim=1)
            mask_upsampled = mask_upsampled.expand(-1, 3, -1, -1)

            

            # #Save images every iteration
            # probabilities = torch.nn.functional.softmax(logits, dim=-1)
            # reconstructed_indices = torch.argmax(probabilities, dim=-1)
            reconstructed_indices = sampled_ids
            # Correctly replace masked indices in reconstructed_indices with 1023
            # print(reconstructed_indices)
            # reconstructed_indices = torch.where(token_indices == 2024, torch.tensor(0, device=token_indices.device), reconstructed_indices)
            # print(reconstructed_indices)

            z_q = model.vqgan.quantize.get_codebook_entry(reconstructed_indices, shape=(batch_size, 16, 16, codebook_emb_dim))
            gen_images_batch = model.vqgan.decode(z_q)

            # Assuming gen_images_batch is the tensor [batch_size, 3, 256, 256] from VQGAN's decode
            # Set pixels to black (0) where mask_upsampled is True
            gen_images_batch[mask_upsampled] = 0

            # Save images
            for b_id in range(batch_size):
                # Apply inverse normalization
                # inv_gen_img = inv_normalize(gen_images_batch[b_id])
                inv_gen_img=gen_images_batch[b_id]
                # inv_orig_img = inv_normalize(orig_images_batch[b_id])

                # Convert to numpy and save - Generated Image
                gen_img_np = np.clip(inv_gen_img.cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                gen_img_np = cv2.cvtColor(gen_img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(b_id + 100*step).zfill(4))), gen_img_np)

        if(per_iter):
            batch_size=args.batch_size

            # #Save images every iteration
            # probabilities = torch.nn.functional.softmax(logits, dim=-1)
            # reconstructed_indices = torch.argmax(probabilities, dim=-1)
            # Correctly replace masked indices in reconstructed_indices with 1023
            # print(reconstructed_indices)
            # print(reconstructed_indices)
            reconstructed_indices = sampled_ids
            z_q = model.vqgan.quantize.get_codebook_entry(reconstructed_indices, shape=(batch_size, 16, 16, codebook_emb_dim))
            gen_images_batch = model.vqgan.decode(z_q)

            # Save images
            for b_id in range(batch_size):
                # Apply inverse normalization
                # inv_gen_img = inv_normalize(gen_images_batch[b_id])
                inv_gen_img=gen_images_batch[b_id]
                # inv_orig_img = inv_normalize(orig_images_batch[b_id])

                # Convert to numpy and save - Generated Image
                gen_img_np = np.clip(inv_gen_img.cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                gen_img_np = cv2.cvtColor(gen_img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(b_id + 1000000*step).zfill(5))), gen_img_np)
        

    # vqgan visualization
    z_q = model.vqgan.quantize.get_codebook_entry(sampled_ids, shape=(bsz, 16, 16, codebook_emb_dim))
    gen_images = model.vqgan.decode(z_q)
    return gen_images


parser = argparse.ArgumentParser('MAGE generation', add_help=False)
parser.add_argument('--temp', default=4.5, type=float,
                    help='sampling temperature')
parser.add_argument('--num_iter', default=12, type=int,
                    help='number of iterations for generation')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size for generation')
parser.add_argument('--num_images', default=50000, type=int,
                    help='number of images to generate')
parser.add_argument('--ckpt', type=str,
                    help='checkpoint')
parser.add_argument('--model', default='mage_vit_base_patch16', type=str,
                    help='model')
parser.add_argument('--output_dir', default='output_dir/fid/gen/mage-vitb', type=str,
                    help='name')
parser.add_argument('--data_path', default='None', type=str,
                    help='name')
parser.add_argument('--dataset', default='coco', type=str,
                    help='dataset name')
parser.add_argument('--vqgan_jax_strongaug', default='vqgan_jax_strongaug.ckpt', type=str,
                    help='dataset name')
parser.add_argument('--log_dir', default='./output_dir',
                    help='path where to tensorboard log')



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

parser.add_argument('--use_decs', default=None,type=int,
                help='2 decoders used')
parser.add_argument('--cross_attn', default=None,type=int,
                help='cross attention on decoder')
parser.add_argument('--slot_vis', default=None,type=int,
                help='slot_vis on decoder')
parser.add_argument('--both_mboi', default=None,type=int,
                help='both_mboi logs decoder')

parser.add_argument('--kmeans_path',  type=str, default='none', help='Kmeans joblib path')





                    
torch.manual_seed(0)
args = parser.parse_args()
args.log_dir=args.output_dir
log_writer = SummaryWriter(log_dir=args.log_dir)


vqgan_ckpt_path = args.vqgan_jax_strongaug



model = models_mage_2dec.__dict__[args.model](norm_pix_loss=False,
                                         mask_ratio_mu=0.55, mask_ratio_std=0.25,
                                         mask_ratio_min=0.0, mask_ratio_max=1.0,
                                         vqgan_ckpt_path=vqgan_ckpt_path,args=args)

# model = models_mage.__dict__[args.model](norm_pix_loss=False,
#                                          mask_ratio_mu=0.55, mask_ratio_std=0.25,
#                                          mask_ratio_min=0.0, mask_ratio_max=1.0,
#                                          vqgan_ckpt_path=vqgan_ckpt_path)
model.to(0)

checkpoint = torch.load(args.ckpt, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

num_steps = args.num_images // args.batch_size + 1
gen_img_list = []
save_folder = os.path.join(args.output_dir, "temp{}-iter{}".format(args.temp, args.num_iter))
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


val_sampler = None

if args.dataset == 'coco':
  val_dataset = COCO2017(root=args.data_path, split='val', image_size=256, mask_size=256,normalization=False)
  val_loader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, shuffle=False, drop_last=False, batch_size=args.batch_size, pin_memory=True,num_workers= 4)#,collate_fn=custom_collate_fn)


else:
    transform_train = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
        # transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor()])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_train)
    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_train = None
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


# Assuming args.dataset is defined somewhere in your code
if args.dataset == 'coco':
    iterator = enumerate(tqdm(val_loader))
else:
    iterator = enumerate(tqdm(data_loader_train))
counter=0
for batch, data in iterator:
    if args.dataset == 'coco':
        image, true_mask_i, true_mask_c, mask_ignore = data
    else:
        image, _ = data

    with torch.no_grad():
        gen_images_batch = gen_image(model=model,image=image, bsz=args.batch_size, seed=batch, choice_temperature=args.temp, num_iter=args.num_iter, data_used=args.dataset,slot_vis=args.slot_vis)
        gen_images_batch = gen_images_batch.detach().cpu()
        gen_img_list.append(gen_images_batch)

        orig_images_batch=image.detach().cpu()
        # save img
        for b_id in range(args.batch_size):

            gen_img = np.clip(gen_images_batch[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255)
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(batch*args.batch_size + b_id).zfill(5))), gen_img)



            inv_orig_img = orig_images_batch[b_id]
            orig_img_np = np.clip(inv_orig_img.numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            orig_img_np = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.output_dir, 'orig_{}.png'.format(str(batch*args.batch_size + b_id).zfill(5))), orig_img_np)
    if batch >0:
        break

log_writer.close()
