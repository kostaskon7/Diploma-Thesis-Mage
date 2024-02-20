from utils_spot import *
from slot_attn import SlotAttentionEncoder
from transformer import TransformerDecoder
from mlp import MlpDecoder
import torch
import random
import math
from functools import partial
from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp




class SPOT(nn.Module):
    def __init__(self, encoder, args, second_encoder=None):
        super().__init__()

        self.which_encoder = args.which_encoder
        self.encoder = encoder
        self.second_encoder = second_encoder
        self.encoder_final_norm = args.encoder_final_norm

        self.use_token_embs = args.use_token_embs
        self.use_token_inds_target = args.use_token_inds_target


        
        for param_name, param in self.encoder.named_parameters():
            if ('blocks' in param_name):
                block_id = int(param_name.split('.')[1])
                if block_id >= args.finetune_blocks_after:
                    param.requires_grad = True  # update by gradient
                else:
                    param.requires_grad = False  # not update by gradient
            else:
                param.requires_grad = False  # not update by gradient
            
        if self.second_encoder is not None:
            for param in self.second_encoder.parameters():
                param.requires_grad = False  # not update by gradient

        # Estimate number of tokens for images of size args.image_size and
        # embedding size (d_model)
        with torch.no_grad():
            x = torch.rand(1, args.img_channels, args.image_size, args.image_size)
            x,_,_ = self.forward_encoder(x, self.encoder)
            _, num_tokens, d_model = x.shape

        # args.d_model = d_model
            ##Allagh edwwwwwwww

        self.num_slots = args.num_slots
        self.d_model = args.d_model



        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size, args.pos_channels,
            args.truncate, args.init_method)

        self.input_proj = nn.Sequential(
            linear(args.d_model, args.d_model, bias=False),
            nn.LayerNorm(args.d_model),
        )



        # --------------------------------------------------------------------------
        # MAGE decoder specifics

        embed_dim=1024
        decoder_embed_dim=512
        patch_size=16
        decoder_num_heads=16
        mlp_ratio=4
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        dropout_rate = 0.1
        decoder_depth=8
        in_chans=3
        vocab_size = self.encoder.codebook_size + 1000 + 1
        self.patch_embed = PatchEmbed(args.image_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches


        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = True

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))  # learnable pos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)

        # --------------------------------------------------------------------------



        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.loss_lambda = args.loss_lambda

        # --------------------------------------------------------------------------


        
        size = int(math.sqrt(num_tokens))
        standard_order = torch.arange(size**2) # This is the default "left_top"
        
        self.cappa = args.cappa
        self.train_permutations = args.train_permutations
        
        if self.train_permutations == 'standard':
            self.permutations = [standard_order]
            self.eval_permutations = 'standard'
        
        else:
            standard_order_2d = standard_order.reshape(size,size)
            
            perm_top_left = torch.tensor([standard_order_2d[row,col] for col in range(0, size, 1) for row in range(0, size, 1)])
            
            perm_top_right = torch.tensor([standard_order_2d[row,col] for col in range(size-1, -1, -1) for row in range(0, size, 1)])
            perm_right_top = torch.tensor([standard_order_2d[row,col] for row in range(0, size, 1) for col in range(size-1, -1, -1)])
            
            perm_bottom_right = torch.tensor([standard_order_2d[row,col] for col in range(size-1, -1, -1) for row in range(size-1, -1, -1)])
            perm_right_bottom = torch.tensor([standard_order_2d[row,col] for row in range(size-1, -1, -1) for col in range(size-1, -1, -1)])
            
            perm_bottom_left = torch.tensor([standard_order_2d[row,col] for col in range(0, size, 1) for row in range(size-1, -1, -1)])
            perm_left_bottom = torch.tensor([standard_order_2d[row,col] for row in range(size-1, -1, -1) for col in range(0, size, 1)])
            
            perm_spiral = spiral_pattern(standard_order_2d, how = 'top_right')
            perm_spiral = torch.tensor((perm_spiral[::-1]).copy())
    
            self.permutations = [standard_order, # left_top
                                 perm_top_left, 
                                 perm_top_right, 
                                 perm_right_top, 
                                 perm_bottom_right, 
                                 perm_right_bottom,
                                 perm_bottom_left,
                                 perm_left_bottom,
                                 perm_spiral
                                 ]
            self.eval_permutations = args.eval_permutations

        self.perm_ind = list(range(len(self.permutations)))

        self.bos_tokens = nn.Parameter(torch.zeros(len(self.permutations), 1, 1, args.d_model))
        torch.nn.init.normal_(self.bos_tokens, std=.02)
        
        self.dec_type = args.dec_type
        self.use_slot_proj = args.use_slot_proj
        
        if self.dec_type=='mlp' and not self.use_slot_proj:
            self.slot_proj = nn.Identity()
            self.dec_input_dim = args.slot_size
        else:
            self.slot_proj = nn.Sequential(
                linear(args.slot_size, args.d_model, bias=False),
                nn.LayerNorm(args.d_model),
            )
            self.dec_input_dim = args.d_model
        
        if self.dec_type=='transformer':
            self.dec = TransformerDecoder(
                args.num_dec_blocks, args.max_tokens, args.d_model, args.num_heads, args.dropout, args.num_cross_heads)
            if self.use_token_inds_target:
                self.dec_predictor = nn.Linear(self.d_model, self.encoder.codebook_size)
            if self.cappa > 0:
                assert (self.train_permutations == 'standard') and (self.eval_permutations == 'standard')   
                self.mask_token = nn.Parameter(torch.zeros(1, 1, args.d_model))
                self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, args.d_model))
                torch.nn.init.normal_(self.pos_embed, std=.02)
                torch.nn.init.normal_(self.mask_token, std=.02)
                  
        elif self.dec_type=='mlp':
            self.dec = MlpDecoder(self.dec_input_dim, args.d_model, args.max_tokens, args.mlp_dec_hidden)

            assert (self.train_permutations == 'standard') and (self.eval_permutations == 'standard')  
        else:
            raise

        if self.dec_type=='transformer':
            # Register hook for capturing the cross-attention (of the query patch
            # tokens over the key/value slot tokens) from the last decoder
            # transformer block of the decoder.
            self.dec_slots_attns = []
            def hook_fn_forward_attn(module, input):
                self.dec_slots_attns.append(input[0])
            self.remove_handle = self.dec._modules["blocks"][-1]._modules["encoder_decoder_attn"]._modules["attn_dropout"].register_forward_pre_hook(hook_fn_forward_attn)


    def forward_encoder(self, x, encoder):
        # tokenization
        with torch.no_grad():
            z_q, _, token_tuple = encoder.vqgan.encode(x)

        _, _, token_indices = token_tuple
        token_indices = token_indices.reshape(z_q.size(0), -1)

        # concate class token
        encoder = encoder.cuda()
        token_indices = token_indices.cuda()
        token_indices = torch.cat(
            [torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = encoder.fake_class_label#Na to dw auto
        token_indices = token_indices.long()
        # bert embedding
        x = encoder.token_emb(token_indices)
        token_emb = x

        for blk in encoder.blocks:
            x = blk(x)
        
        # return x[:,1:,:],token_emb[:,1:,:],token_indices[:,1:]
        return x,token_emb,token_indices
    
    def forward_encoder_mage(self, x,encoder):
        # tokenization
        with torch.no_grad():
            z_q, _, token_tuple = encoder.vqgan.encode(x)

        _, _, token_indices = token_tuple
        token_indices = token_indices.reshape(z_q.size(0), -1)
        gt_indices = token_indices.clone().detach().long()

        # masking
        bsz, seq_len = token_indices.size()
        mask_ratio_min = encoder.mask_ratio_min
        mask_rate = encoder.mask_ratio_generator.rvs(1)[0]

        num_dropped_tokens = int(np.ceil(seq_len * mask_ratio_min))
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))

        # it is possible that two elements of the noise is the same, so do a while loop to avoid it
        while True:
            noise = torch.rand(bsz, seq_len, device=x.device)  # noise in [0, 1]
            sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
            cutoff_drop = sorted_noise[:, num_dropped_tokens-1:num_dropped_tokens]
            cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
            token_drop_mask = (noise <= cutoff_drop).float()
            token_all_mask = (noise <= cutoff_mask).float()
            if token_drop_mask.sum() == bsz*num_dropped_tokens and token_all_mask.sum() == bsz*num_masked_tokens:
                break
            else:
                print("Rerandom the noise!")
        # print(mask_rate, num_dropped_tokens, num_masked_tokens, token_drop_mask.sum(dim=1), token_all_mask.sum(dim=1))
        token_indices[token_all_mask.nonzero(as_tuple=True)] = encoder.mask_token_label
        # print("Masekd num token:", torch.sum(token_indices == self.mask_token_label, dim=1))

        # concate class token
        token_indices = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = encoder.fake_class_label
        token_drop_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_drop_mask], dim=1)
        token_all_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_all_mask], dim=1)
        token_indices = token_indices.long()
        # bert embedding
        input_embeddings = encoder.token_emb(token_indices)
        # print("Input embedding shape:", input_embeddings.shape)
        bsz, seq_len, emb_dim = input_embeddings.shape

        # dropping
        token_keep_mask = 1 - token_drop_mask
        input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)
        # print("Input embedding after drop shape:", input_embeddings_after_drop.shape)

        # apply Transformer blocks
        x = input_embeddings_after_drop
        for blk in encoder.blocks:
            x = blk(x)
        x = encoder.norm(x)
        # print("Encoder representation shape:", x.shape)

        return x, gt_indices, token_drop_mask, token_all_mask


    def forward_decoder(self, slots, emb_target):
        # Prepate the input tokens for the decoder transformer:
        # (1) insert a learnable beggining-of-sequence ([BOS]) token at the beggining of each target embedding sequence.
        # (2) remove the last token of the target embedding sequence
        # (3) no need to add positional embeddings since positional information already exists at the DINO's outptu.
        

        if self.training:
            if self.train_permutations == 'standard':
                which_permutations = [0] # USE [0] FOR THE STANDARD ORDER
            elif self.train_permutations == 'random':
                which_permutations = [random.choice(self.perm_ind)]
            elif self.train_permutations == 'all':
                which_permutations = self.perm_ind
            else:
                raise
        else:
            if self.eval_permutations == 'standard':
                which_permutations = [0] # USE [0] FOR THE STANDARD ORDER
            elif self.eval_permutations == 'random':
                which_permutations = [random.choice(self.perm_ind)]
            elif self.eval_permutations == 'all':
                which_permutations = self.perm_ind
            else:
                raise
        
        all_dec_slots_attns = []
        all_dec_output = []
        for perm_id in which_permutations:
            current_perm = self.permutations[perm_id]

            bos_token = self.bos_tokens[perm_id]
            bos_token = bos_token.expand(emb_target.shape[0], -1, -1)
            
            use_pos_emb = self.cappa > 0
            parallel_dec = self.cappa > 0 and ((self.cappa >= 1.0) or (self.training and random.random() < self.cappa))
            #print(f"Paralled Decoder (CAPPA) {parallel_dec}")
            # Input to the decoder
            if parallel_dec: # Use parallel decoder
                dec_input = self.mask_token.to(emb_target.dtype).expand(emb_target.shape[0], -1, -1)
            else: # Use autoregressive decoder
                # first_element = [p for p in current_perm if p == 0]
                # filtered_perm = [p for p in current_perm if p != 0]
                # dec_input = torch.cat((emb_target[:, first_element , :], emb_target[:, filtered_perm, :]), dim=1)

                # dec_input = emb_target[:, :-1 , :]
                # print(emb_target)

                # dec_input = torch.cat((bos_token,emb_target[:, first_element , :],emb_target[:,1:,:][:, filtered_perm , :]), dim=1)
                dec_input = torch.cat((bos_token, emb_target[:,current_perm,:][:,:-1,:]), dim=1)

            if use_pos_emb:
                # Add position embedding if they exist.
                dec_input = dec_input + self.pos_embed.to(emb_target.dtype)

            # dec_input has the same shape as emb_target, which is [B, N, D]
            dec_input = self.input_proj(dec_input)
    
            # Apply the decoder
            dec_input_slots = self.slot_proj(slots) # shape: [B, num_slots, D]

            if self.dec_type=='transformer':
                dec_output = self.dec(dec_input, dec_input_slots, causal_mask=(not parallel_dec))
                # decoder_output shape [B, N, D]

                dec_slots_attns = self.dec_slots_attns[0]
                self.dec_slots_attns = []

                # sum over the heads and 
                dec_slots_attns = dec_slots_attns.sum(dim=1) # [B, N, num_slots]
                # dec_slots_attns shape [B, num_heads, N, num_slots]
                # L1-normalize over the slots so as to sum to 1.
                dec_slots_attns = dec_slots_attns / dec_slots_attns.sum(dim=2, keepdim=True)
                inv_current_perm = torch.argsort(current_perm)


                dec_slots_attns = dec_slots_attns[:,inv_current_perm,:]
                dec_output = dec_output[:,inv_current_perm,:]

            elif self.dec_type=='mlp':
                dec_output, dec_slots_attns = self.dec(dec_input_slots)
                dec_slots_attns = dec_slots_attns.transpose(1,2)

            else:
                raise
            
            all_dec_slots_attns.append(dec_slots_attns)
            all_dec_output.append(dec_output)


        mean_dec_slots_attns = torch.stack(all_dec_slots_attns).mean(0)
        mean_dec_output = torch.stack(all_dec_output).mean(0)


        return mean_dec_output, mean_dec_slots_attns
    


    def forward_decoder_mage(self, x, token_drop_mask, token_all_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        if self.pad_with_cls_token:
            mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)
        else:
            mask_tokens = self.mask_token.repeat(token_all_mask.shape[0], token_all_mask.shape[1], 1)

        # put undropped tokens into original sequence
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # set undropped but masked positions with mask
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)

        # add pos embed
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)
        # print("Logits shape:", x.shape)

        return x
    

    def forward_decoder_generation(self, slots, n_tokens=256):
        # Prepate the input tokens for the decoder transformer:
        # (1) insert a learnable beggining-of-sequence ([BOS]) token at the beggining of each target embedding sequence.
        # (2) remove the last token of the target embedding sequence
        # (3) no need to add positional embeddings since positional information already exists at the DINO's outptu.
        

        bos_token = self.bos_tokens[0]
        bos_token = bos_token.expand(slots.shape[0], -1, -1)
        



        # dec_input = torch.cat((bos_token, emb_target[:,:,:][:, :-1, :]), dim=1)


        dec_input = bos_token
        dec_input_slots = self.slot_proj(slots) # shape: [B, num_slots, D]
        for i in range(n_tokens):


            # dec_input has the same shape as emb_target, which is [B, N, D]
            dec_input_proj = self.input_proj(dec_input)


            # Apply the decoder
            dec_output = self.dec(dec_input_proj, dec_input_slots, causal_mask=True)
            # decoder_output shape [B, N, D]


            dec_input = torch.cat((dec_input, dec_output[:, -1, :].unsqueeze(1)), dim=1)



            dec_slots_attns = self.dec_slots_attns[0]
            self.dec_slots_attns = []

            # sum over the heads and 
            dec_slots_attns = dec_slots_attns.sum(dim=1) # [B, N, num_slots]
            # dec_slots_attns shape [B, num_heads, N, num_slots]
            # L1-normalize over the slots so as to sum to 1.
            dec_slots_attns = dec_slots_attns / dec_slots_attns.sum(dim=2, keepdim=True)


        return dec_output, dec_slots_attns
    

    def forward_loss_mage(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        # logits and mask are with seq_len+1 but gt_indices is with seq_len
        loss = self.criterion(logits[:, 8:, :self.codebook_size].reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))#DEN EIMAI SIGOUROS GIA TO +1 H +7

        # print(loss.shape)
        loss = loss.reshape(bsz, seq_len)
        # print(loss.shape)
        loss = (loss * mask[:, 1:]).sum() / mask[:, 1:].sum()  # mean loss on removed patches
        # print(loss)
        # print("Telos")
        return loss
    

    def get_embeddings_n_slots(self, image):
        """
        image: batch_size x img_channels x H x W
        """

        B, _, H, W = image.size()
        with torch.no_grad():
            emb_target,_,_ = self.forward_encoder(image, self.encoder)
        # emb_target shape: B, N, D

        # Apply the slot attention
        slots, slots_attns, _ = self.slot_attn(emb_target)
        return emb_target, slots, slots_attns

    def forward(self, image,gen=False):
        """
        image: batch_size x img_channels x H x W
        """

        B, _, H, W = image.size()
        with torch.no_grad():
            latent_mask, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder_mage(image, self.encoder)

            emb_input, token_emb, token_indices = self.forward_encoder(image, self.encoder)

        

        if self.use_token_embs:
            emb_target = token_emb.clone().detach()
        else:
            emb_target = emb_input.clone().detach()
        # emb_target shape: B, N, D
        # print(emb_target.shape)

        # Apply the slot attention
        slots, slots_attns, init_slots, attn_logits = self.slot_attn(emb_input)

        # slots, slots_attns, init_slots, attn_logits = self.slot_attn(emb_target)

        attn_logits = attn_logits.squeeze()
        # slots shape: [B, num_slots, Ds]
        # slots_attns shape: [B, N, num_slots]

        # Apply the decoder.
        if gen:
            dec_recon, dec_slots_attns = self.forward_decoder_generation(slots)
        else :
            # dec_recon, dec_slots_attns = self.forward_decoder(slots, emb_target[:, 1:, :])
            dec_recon, dec_slots_attns = self.forward_decoder(slots, emb_target)
            logits = self.forward_decoder_mage(latent_mask,slots ,token_drop_mask, token_all_mask)
            self.dec_preds=logits




        # dec_recon, dec_slots_attns = self.forward_decoder_generation(slots )

        # Mean-Square-Error loss
        H_enc, W_enc = int(math.sqrt(emb_target.shape[1])), int(math.sqrt(emb_target.shape[1]))

        # torch.Size([64, 256, 768])
        # torch.Size([64, 256, 768])
        if self.use_token_inds_target:
            # if self.training:
            #     dec_preds =self.dec_predictor(dec_recon)
            # else :
            dec_preds =self.dec_predictor(dec_recon)

            self.dec_preds=dec_preds
            # token_indices = token_indices.reshape(-1)


            token_indices = token_indices[:,1:].reshape(-1)
            dec_preds = dec_preds.reshape(-1, dec_preds.shape[2])
            loss = nn.CrossEntropyLoss()

            loss_out = loss(dec_preds,token_indices)
        else:
            # loss_out = ((emb_target - dec_recon) ** 2).sum()/(B*H_enc*W_enc*self.d_model)
            loss_out = ((emb_target[:,1:,:] - dec_recon) ** 2).sum()/(B*H_enc*W_enc*self.d_model)# changed emb_target shape

            loss_mage = forward_loss_mage(gt_indices, logits, token_all_mask)

            
        # Reshape the slot and decoder-slot attentions.
        # slots_attns = slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)
        slots_attns = slots_attns[:,1:,:].transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

        dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

        


        return loss_out,loss_mage, slots_attns, dec_slots_attns, slots, dec_recon, attn_logits




class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale

        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss