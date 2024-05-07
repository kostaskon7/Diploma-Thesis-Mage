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
from torch_kmeans import KMeans
from kmeans_pytorch import kmeans
from sklearn.cluster import MiniBatchKMeans
from joblib import dump, load
import os









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

parser.add_argument('--both_mboi', default=None,type=int,
                help='cross attention on decoder')



                    

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
  val_dataset = COCO2017(root=args.data_path, split='train', image_size=256, mask_size=256)
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
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

collected_outputs = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming args.dataset is defined somewhere in your code

counter=0
for batch, image in enumerate(tqdm(val_loader, desc="Processing images")):


    image=image.cuda()
    with torch.no_grad():
        # val_loss, _, _, default_slots_attns, _, _ = model(image)
        latent= model.forward_encoder(image)
        #slots, attn, init_slots, attn_logits = self.slot_attention(latent[:,1:,:])
        latent=latent[:,1:,:]

        slots, attn, _, _ = model.slot_attention(latent)

        # slots=model.slot_proj2(slots)

        # slots, attn, attn_logits = model.masked_trans(latent,(16, 16))

        


        # # Hard Mask pooling
        # attn=attn.clone().detach()
        # attn_onehot = torch.nn.functional.one_hot(attn.argmax(2), num_classes=model.slot_attention.num_slots).to(latent.dtype)
        # # To add normalization
        # # attn_onehot = attn_onehot / torch.sum(attn_onehot+self.epsilon, dim=-2, keepdim=True)
        # slots_pool = torch.matmul(attn_onehot.transpose(-1, -2), latent)
        # slots=model.slot_proj2(slots_pool)

        attn=attn.clone().detach()
        # Latent another transformation?
        attn_onehot = torch.nn.functional.one_hot(attn.argmax(2), num_classes=7).to(latent.dtype)
        slots_pool = torch.matmul(attn_onehot.transpose(-1, -2), latent)


        # slots_pool = torch.matmul(attn.transpose(-1, -2), latent)

        slots=model.slot_proj2(slots_pool)
        collected_outputs.append(slots)
    


# all_slots = torch.cat(collected_outputs, dim=0)


# all_slots_reshaped = all_slots.view(-1,7, 256)
# # Now all_outputs is [total_images, 7, 256], directly ready for KMeans without additional reshaping
# all_slots_reshaped=all_slots_reshaped.cuda()


# total_elements = all_slots.numel()

# Since the last dimension is fixed at 256, calculate the middle dimension dynamically
# Note: 'total_elements // 256' gives the total size for the first two dimensions combined
# The new shape's middle dimension is calculated by dividing the total size by the number of batches and then by 256
# new_middle_dimension = (total_elements // 256) // all_slots.shape[0]

# Reshape the combined tensor to the new shape
# reshaped_slots = all_slots.reshape(all_slots.shape[0], new_middle_dimension, 256)



# Make sure your KMeans supports GPU, and `all_outputs_reshape` is on the right device
# kmeans_model = kmeans(num_classes=81, mode='euclidean', verbose=1, device=device)
# labels = kmeans_model.fit(all_slots_reshaped)

# cluster_ids_x, cluster_centers = kmeans(
#     X=all_slots_reshaped, num_clusters=1024, distance='euclidean', device=device,tol=6e-3
# )

# model = KMeans(n_clusters=1024,tol=6e-3)
# model = model.fit(reshaped_slots)



## MINIBATCH SKLEARN

# Step 1: Concatenate all collected outputs
all_slots_tensor = torch.cat(collected_outputs, dim=0)

# Step 2: Reshape the tensor to 2D [number_of_samples, 256]
# Since each original tensor is [batch_size, 7, 256], and you're concatenating along the batch dimension,
# you can simply reshape it to (-1, 256) to flatten all but the last dimension.
data_2d = all_slots_tensor.reshape(-1, 768)

# Step 3: Convert to NumPy array if you're using PyTorch
data_2d_np = data_2d.cpu().numpy()

# Step 4: Apply MiniBatchKMeans
n_clusters = 1024  # Example: Define the number of clusters
kmeans = MiniBatchKMeans(n_clusters=n_clusters,tol=0.0001)  # Adjust batch_size as necessary
kmeans.fit(data_2d_np)



directory = '/data/kmeans/hard_100/'
file_name = 'kmeans_model1024_100ep_hard.joblib'

full_path = os.path.join(directory, file_name)


# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)


dump(kmeans, file_name)

n_clusters = 2048  # Example: Define the number of clusters
kmeans = MiniBatchKMeans(n_clusters=n_clusters,tol=0.0001)  # Adjust batch_size as necessary
kmeans.fit(data_2d_np)



directory = '/data/kmeans/hard_100/'
file_name = 'kmeans_model2048_100ep_hard.joblib'

full_path = os.path.join(directory, file_name)


# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

dump(kmeans, full_path)


n_clusters = 4096  # Example: Define the number of clusters
kmeans = MiniBatchKMeans(n_clusters=n_clusters,tol=0.0001)  # Adjust batch_size as necessary
kmeans.fit(data_2d_np)


directory = '/data/kmeans/hard_100/'
file_name = 'kmeans_model4096_100ep_hard.joblib'

full_path = os.path.join(directory, file_name)


# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

dump(kmeans, full_path)


n_clusters = 8192  # Example: Define the number of clusters
kmeans = MiniBatchKMeans(n_clusters=n_clusters,tol=0.0001)  # Adjust batch_size as necessary
kmeans.fit(data_2d_np)


directory = '/data/kmeans/hard_100/'
file_name = 'kmeans_model8192_100ep_hard.joblib'

full_path = os.path.join(directory, file_name)


# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

dump(kmeans, full_path)

n_clusters = 16384  # Example: Define the number of clusters
kmeans = MiniBatchKMeans(n_clusters=n_clusters,tol=0.0001)  # Adjust batch_size as necessary
kmeans.fit(data_2d_np)


directory = '/data/kmeans/hard_100/'
file_name = 'kmeans_model16384_100ep_hard.joblib'

full_path = os.path.join(directory, file_name)


# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

dump(kmeans, full_path)


# kmeans = load('kmeans_model.joblib')







# Save your model
# torch.save(model, 'cluster_centers_1024.pth')