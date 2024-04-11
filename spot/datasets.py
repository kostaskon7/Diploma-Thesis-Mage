import os
import glob
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from pycocotools import mask
from pycocotools.coco import COCO

from skimage import measure

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PascalVOC(Dataset):
    def __init__(self, root, split, image_size=224, mask_size = 224):
        assert split in ['trainaug', 'val']
        imglist_fp = os.path.join(root, 'ImageSets/Segmentation', split+'.txt')
        self.imglist = self.read_imglist(imglist_fp)

        self.root = root
        self.train_transform = transforms.Compose([
                            transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.RandomCrop(image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])

        self.val_transform_image = transforms.Compose([transforms.Resize(size = image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                               transforms.CenterCrop(size = image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.val_transform_mask = transforms.Compose([transforms.Resize(size = mask_size, interpolation=transforms.InterpolationMode.NEAREST),
                               transforms.CenterCrop(size = mask_size),
                               transforms.PILToTensor()])
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size

    def __getitem__(self, idx):

        imgname = self.imglist[idx]
        img_fp = os.path.join(self.root, 'JPEGImages', imgname) + '.jpg'
        mask_fp_class = os.path.join(self.root, 'SegmentationClass', imgname) + '.png'
        mask_fp_instance = os.path.join(self.root, 'SegmentationObject', imgname) + '.png'

        img = Image.open(img_fp)

        if self.split=='trainaug':
            
            img = self.train_transform(img)
            
            return img
   
        elif self.split=='val':
            
            mask_class    = Image.open(mask_fp_class)
            mask_instance = Image.open(mask_fp_instance)
            
            img = self.val_transform_image(img)
            
            mask_class = self.val_transform_mask(mask_class).squeeze().long()
            mask_class[mask_class==255]=0 # Ignore objects' boundaries

            mask_instance = self.val_transform_mask(mask_instance).squeeze().long()
            mask_instance[mask_instance==255]=0 # Ignore objects' boundaries
            
            ignore_mask = torch.zeros((1,self.mask_size,self.mask_size), dtype=torch.long) # There is no overlapping in VOC

            return img, mask_instance, mask_class, ignore_mask
        
        else:
            
            mask_class    = Image.open(mask_fp_class)
            mask_instance = Image.open(mask_fp_instance)
            
            return img, mask_instance.long(), mask_instance.squeeze()


    def __len__(self):
        return len(self.imglist)

    def read_imglist(self, imglist_fp):
        ll = []
        with open(imglist_fp, 'r') as fd:
            for line in fd:
                ll.append(line.strip())
        return ll


class COCO2017(Dataset):
    NUM_CLASSES = 81
    CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 
 89, 90]
    
    assert(NUM_CLASSES) == len(set(CAT_LIST))

    def __init__(self, root, split='train', year='2017', image_size=224, mask_size=224, return_gt_in_train=False):
        super().__init__()
        ann_file = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, year))
        self.img_dir = os.path.join(root, '{}{}'.format(split, year))
        if not os.path.isdir(self.img_dir):
            self.img_dir = os.path.join(root, "images", '{}{}'.format(split, year))
            assert os.path.isdir(self.img_dir)
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        self.return_gt_in_train = return_gt_in_train

        self.ids = list(self.coco.imgs.keys())
        
        self.train_transform = transforms.Compose([
                            transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                        ])
                        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        # ])
        
        self.val_transform_image = transforms.Compose([transforms.Resize(size = image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                               transforms.CenterCrop(size = image_size),
                               transforms.ToTensor()])
                            #    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            #    ])

        self.val_transform_mask = transforms.Compose([transforms.Resize(size = mask_size, interpolation=transforms.InterpolationMode.NEAREST),
                               transforms.CenterCrop(size = mask_size),
                               transforms.PILToTensor()])
        self.image_size = image_size

    def __getitem__(self, index):
        img, mask_instance, mask_class, mask_ignore = self._make_img_gt_point_pair(index)

        if self.split == "train" and (self.return_gt_in_train is False):
            
            img = self.train_transform(img)
            
            return img
        elif self.split == "train" and (self.return_gt_in_train is True):
            img = self.val_transform_image(img)
            mask_class = self.val_transform_mask(mask_class)
            mask_instance = self.val_transform_mask(mask_instance)
            mask_ignore = self.val_transform_mask(mask_ignore)

            if random.random() < 0.5:
                img = TF.hflip(img)
                mask_class = TF.hflip(mask_class)
                mask_instance = TF.hflip(mask_instance)
                mask_ignore = TF.hflip(mask_ignore)
            
            mask_class = mask_class.squeeze().long()
            mask_instance = mask_instance.squeeze().long()
            mask_ignore = mask_ignore.squeeze().long()

            return img, mask_instance, mask_class, mask_ignore        
        elif self.split =='val':

            img = self.val_transform_image(img)
            mask_class = self.val_transform_mask(mask_class).squeeze().long()
            mask_instance = self.val_transform_mask(mask_instance).squeeze().long()
            mask_ignore = self.val_transform_mask(mask_ignore).squeeze().long().unsqueeze(0)
            
            return img, mask_instance, mask_class, mask_ignore
        else:
            raise

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _targets = self._gen_seg_n_insta_masks(cocotarget, img_metadata['height'], img_metadata['width'])
        mask_class = Image.fromarray(_targets[0])
        mask_instance = Image.fromarray(_targets[1])
        mask_ignore = Image.fromarray(_targets[2])
        return _img, mask_instance, mask_class, mask_ignore

    def _gen_seg_n_insta_masks(self, target, h, w):
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        insta_mask = np.zeros((h, w), dtype=np.uint8)
        ignore_mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for i, instance in enumerate(target, 1):
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                seg_mask[:, :] += (seg_mask == 0) * (m * c)
                insta_mask[:, :] += (insta_mask == 0) * (m * i)
                ignore_mask[:, :] += m
            else:
                seg_mask[:, :] += (seg_mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
                insta_mask[:, :] += (insta_mask == 0) * (((np.sum(m, axis=2)) > 0) * i).astype(np.uint8)
                ignore_mask[:, :] += (((np.sum(m, axis=2)) > 0) * 1).astype(np.uint8)

        # Ignore overlaps
        ignore_mask = (ignore_mask>1).astype(np.uint8)

        all_masks = np.stack([seg_mask, insta_mask, ignore_mask])
        return all_masks

    def __len__(self):
        return len(self.ids)


class MOVi(Dataset):
    def __init__(self, root, split, image_size, mask_size, num_segs=25, frames_per_clip=24, img_glob='*_image.png', predefined_json_paths = None):
        
        self.root = root
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size
        self.total_dirs = sorted(glob.glob(os.path.join(root, '*')))
        self.frames_per_clip = frames_per_clip
        
        if self.split == 'train' and predefined_json_paths is not None:
            with open(predefined_json_paths, 'r') as fp:
                paths_persistence = json.load(fp)
            self.rgb = [Path(p) for p in paths_persistence['rgb']]
            self.mask = [[Path(p) for p in m] for m in paths_persistence['mask']]
            
        else:
            self.rgb = []
            self.mask = []
            for dir in self.total_dirs:
                frame_buffer = []
                mask_buffer = []
                image_paths = glob.glob(os.path.join(dir, img_glob))
                if self.split == 'train':
                    random.shuffle(image_paths)
                    image_paths = image_paths[:self.frames_per_clip]
                else:
                    image_paths = sorted(image_paths)
                for image_path in image_paths:
                    p = Path(image_path)
    
                    frame_buffer.append(p)
                    mask_buffer.append([
                        p.parent / f"{p.stem.split('_')[0]}_mask_{n:02}.png" for n in range(num_segs)
                    ])
    
                self.rgb.extend(frame_buffer)
                self.mask.extend(mask_buffer)
                frame_buffer = []
                mask_buffer = []
            
        if self.split == 'train' and predefined_json_paths is None:
            paths_persistence = dict(rgb=[str(p) for p in self.rgb], mask=[[str(p) for p in m] for m in self.mask])
                    
            with open(self.split+'_movi_paths.json', 'w') as fp:
                json.dump(paths_persistence, fp)
        
        self.train_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), 
                                                                  (0.229, 0.224, 0.225))])
        self.val_transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        
        img_loc = self.rgb[idx]
        img = Image.open(img_loc).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        img = self.train_transform(img)

        if self.split == 'train':
            return img
        else:
            mask_locs = self.mask[idx]
            masks = []
            for mask_loc in mask_locs:
                mask = Image.open(mask_loc).convert('1')
                mask = mask.resize((self.mask_size, self.mask_size))
                mask = self.val_transforms(mask)
                masks += [mask]
            masks = torch.stack(masks, dim=0).squeeze().long()
    
            mask_instance = torch.zeros((self.mask_size,self.mask_size), dtype=torch.long)
            mask_class = torch.zeros((self.mask_size,self.mask_size), dtype=torch.long) # There are no semantic segmentations in MOVi
            ignore_mask = torch.zeros((1,self.mask_size,self.mask_size), dtype=torch.long) # There is no overlapping in MOVi
            
            for i, instance in enumerate(masks):
                mask_instance[:, :] += instance * i

            return img, mask_instance, mask_class, ignore_mask
        

class PartImageNetDataset(Dataset):
    CAT_LIST = [i for i in range(1, 159)]  # Example: 158 classes, adjust as needed

    def __init__(self, root, split='train', image_size=256, mask_size=256):
        """
        Initialize the PartImageNet dataset.

        :param root: Root directory of the PartImageNet dataset.
        :param split: Dataset split ('train', 'val', or 'test').
        :param image_size: Desired size of the output images.
        :param mask_size: Desired size of the output masks.
        """
        self.root_dir = root
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size
        self.images_dir = os.path.join(root, 'images', split)
        self.annotations_dir = os.path.join(root, 'annotations', split)
        self.image_ids = [f.split('.')[0] for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f))]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, f"{img_id}.JPEG")
        mask_path = os.path.join(self.annotations_dir, f"{img_id}.png")

        # Load image
        image = Image.open(img_path).convert('RGB')
        # Resize image
        image = transforms.Resize((self.image_size, self.image_size), Image.BILINEAR)(image)
        # Convert to tensor
        image = transforms.ToTensor()(image)

        # Load mask and resize
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = transforms.Resize((self.mask_size, self.mask_size), Image.NEAREST)(mask)
        mask_array = np.array(mask)

        # Label different objects in the mask
        # labeled_mask, num_features = measure.label(mask_array, background=0, return_num=True)
        labeled_mask, num_features = measure.label(mask_array,  return_num=True)

        # Generate shades of gray for each label
        # shades_of_gray = np.linspace(255, 0, num_features + 1, dtype=np.uint8)[1:]  # Exclude background
        shades_of_gray = np.linspace(255, 0, num_features + 1, dtype=np.uint8)[:]  # NOT Exclude background
        labeled_mask_overlay = np.zeros_like(mask_array, dtype=np.uint8)
        for label, shade in zip(range(1, num_features + 1), shades_of_gray):
            labeled_mask_overlay[labeled_mask == label] = shade

        # Convert the labeled mask back into an Image and then to a tensor
        true_mask_i = transforms.ToTensor()(Image.fromarray(labeled_mask_overlay))

        # Placeholder for true_mask_c and mask_ignore, create as needed
        true_mask_c = transforms.ToTensor()(mask)
        mask_ignore = torch.zeros_like(true_mask_c)  # Placeholder for the ignore mask

        return image, true_mask_i, true_mask_c, mask_ignore