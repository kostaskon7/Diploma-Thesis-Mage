## Clone Repo

git clone https://github.com/kostaskon7/Diploma-Thesis-Mage.git

git checkout testing

## Download mage checkpoints and vqgan

https://drive.google.com/file/d/1Q6tbt3vF0bSrv5sPrjpFu8ksG3vTsVX2/view

https://drive.google.com/file/d/13S_unB87n6KKuuMdyMnyExW0G1kplTbP/view

##  Create conda environment

conda env create -f env_exp.yml --name mage

conda activate mage

## Testing

python main_pretrain_2dec.py --batch_size 16 --model mage_vit_base_patch16 --resume path/to//mage-vitb-1600.pth --warmup_epochs 5 --vqgan_ckpt_path /path/to/vqgan_jax_strongaug.ckpt --epochs 50  --lr 2.e-4 --weight_decay 0.05 --output_dir /data/outputs/ --data_path /data/coco

### Take the path of the helpers.py file, the created an error

Edit file change_torch.py and replace file_path  with your path

./change_torch.py 
