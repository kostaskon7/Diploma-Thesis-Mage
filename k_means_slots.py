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
from sklearn.preprocessing import StandardScaler










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




parser.add_argument('--ce_weight', type=float, default=5e-3, help='weight of the cross-entropy distilation loss')
parser.add_argument('--final_ce_weight', type=float, default=None, help='final weight of the cross-entropy distilation loss')

parser.add_argument('--crf_dir', type=str, default=None, help='Directory of crf files')
parser.add_argument('--directory', type=str, default=None, help='Directory of crf files')

parser.add_argument('--max_iterations',  type=int, default=300, help='Max iterations in kmeans')
parser.add_argument('--tol',  type=float, default=1e-3, help='Max tolerance reached')



def kmeans_plusplus(X, n_clusters, random_state=None):
    if random_state:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    n_samples, _ = X.shape
    centers = torch.empty((n_clusters, X.shape[1]), device=X.device, dtype=X.dtype)
    indices = torch.empty((n_clusters,), dtype=torch.long, device=X.device)  # To store the indices

    # Randomly choose the first center
    first_center_idx = np.random.choice(n_samples)
    centers[0] = X[first_center_idx]
    indices[0] = first_center_idx

    # Initialize a list to store the minimum distances for each point to any center
    closest_dist_sq = torch.full((n_samples,), float('inf'), device=X.device, dtype=X.dtype)

    for i in range(1, n_clusters):
        # Compute the distance from each point to the nearest center
        dist_to_new_center = torch.sum((X - centers[i - 1]) ** 2, dim=1)
        closest_dist_sq = torch.min(closest_dist_sq, dist_to_new_center)

        # Choose the next center with a probability proportional to the squared distance
        probs = closest_dist_sq / torch.sum(closest_dist_sq)
        cumulative_probs = torch.cumsum(probs, dim=0)
        r = torch.rand(1, device=X.device, dtype=X.dtype)
        next_center_idx = torch.searchsorted(cumulative_probs, r).item()
        centers[i] = X[next_center_idx]
        indices[i] = next_center_idx

    return centers, indices



                    

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
msg=model.load_state_dict(checkpoint['model'],strict=False)
print(msg)
model.eval()

num_steps = args.num_images // args.batch_size + 1
gen_img_list = []
save_folder = os.path.join(args.output_dir, "temp{}-iter{}".format(args.temp, args.num_iter))
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


val_sampler = None

if args.dataset == 'coco':
  val_dataset = COCO2017(root=args.data_path, split='train', image_size=256, mask_size=256,normalization=False)
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
        latent,_,_,_= model.forward_encoder_copy(image)
        #slots, attn, init_slots, attn_logits = self.slot_attention(latent[:,1:,:])
        latent=latent[:,1:,:]

        slots, attn, _, _ = model.slot_attention(latent)
        # breakpoint()


        attn=attn.clone().detach()
        # Latent another transformation?
        attn_onehot = torch.nn.functional.one_hot(attn.argmax(2), num_classes=7).to(latent.dtype)
        attn_onehot = attn_onehot / torch.sum(attn_onehot+model.epsilon, dim=-2, keepdim=True)
        # slots = torch.matmul(attn_onehot.transpose(-1, -2), latent)
        slots = torch.matmul(attn_onehot.transpose(-1, -2), latent)


        # breakpoint()
        # slots_pool = torch.matmul(attn.transpose(-1, -2), latent)

        slots=model.slot_proj2(slots)
        collected_outputs.append(slots)
        # break

    

breakpoint()


## MINIBATCH SKLEARN
# breakpoint()

tolerance = args.tol
max_iterations = args.max_iterations

scaler = StandardScaler()



# Step 1: Concatenate all collected outputs
all_slots_tensor = torch.cat(collected_outputs, dim=0)

# Step 2: Reshape the tensor to 2D [number_of_samples, 256]
# Since each original tensor is [batch_size, 7, 256], and you're concatenating along the batch dimension,
# you can simply reshape it to (-1, 256) to flatten all but the last dimension.
# data_2d = all_slots_tensor.reshape(-1, 768)
data_2d = all_slots_tensor.reshape(-1, 768)
num_samples = 10000


breakpoint()

# num_samples = 10000

# # Perform k-means++ initialization
# sampled_centers, sampled_indices = kmeans_plusplus(data_2d, n_clusters=num_samples, random_state=0)





# Step 3: Convert to NumPy array if you're using PyTorch
data_2d_np = data_2d.cpu().numpy()

data_2d_np_normalized = scaler.fit_transform(data_2d_np)




directory = args.directory


n_clusters = 16384  # Example: Define the number of clusters
kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, tol=tolerance, max_iter=max_iterations,max_no_improvement=None)  # Adjust batch_size as necessary
kmeans_model.fit(data_2d_np_normalized)


file_name = 'kmeans_model16384_100ep_hard.joblib'

full_path = os.path.join(directory, file_name)

full_path_scaler = os.path.join(directory, 'scaler.joblib')



# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

dump(kmeans_model, full_path)

dump(scaler, full_path_scaler)

print(f"Number of iterations: {kmeans_model.n_iter_}")
print(f"Tolerance used for stopping criterion: {kmeans_model.tol}")
print(f"inertia_ used for stopping criterion: {kmeans_model.inertia_}")





n_clusters = 32768  # Example: Define the number of clusters
kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, tol=tolerance, max_iter=max_iterations)  # Adjust batch_size as necessary
kmeans_model.fit(data_2d_np_normalized)


file_name = 'kmeans_model32768_100ep_hard.joblib'

full_path = os.path.join(directory, file_name)


# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

dump(kmeans_model, full_path)

print(f"Number of iterations: {kmeans_model.n_iter_}")
print(f"Tolerance used for stopping criterion: {kmeans_model.tol}")

n_clusters = 65536  # Example: Define the number of clusters
kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, tol=tolerance, max_iter=max_iterations)  # Adjust batch_size as necessary
kmeans_model.fit(data_2d_np_normalized)


file_name = 'kmeans_model65536_100ep_hard.joblib'

full_path = os.path.join(directory, file_name)


# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

dump(kmeans_model, full_path)

print(f"Number of iterations: {kmeans_model.n_iter_}")
print(f"Tolerance used for stopping criterion: {kmeans_model.tol}")
