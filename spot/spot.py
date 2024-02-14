from utils_spot import *
from slot_attn import SlotAttentionEncoder
from transformer import TransformerDecoder
from mlp import MlpDecoder
import torch
import random
import math


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

        print(args.d_model)
        print("Edw d_model")

        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size, args.pos_channels,
            args.truncate, args.init_method)

        self.input_proj = nn.Sequential(
            linear(args.d_model, args.d_model, bias=False),
            nn.LayerNorm(args.d_model),
        )
        
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
                first_element = [p for p in current_perm if p == 0]
                filtered_perm = [p for p in current_perm if p != 0]
                # dec_input = torch.cat((emb_target[:, first_element , :], emb_target[:, filtered_perm, :]), dim=1)

                # dec_input = emb_target[:, :-1 , :]
                # print(emb_target)
                print(emb_target[:, first_element , :].shape)
                print(emb_target[:,1:,:][:, filtered_perm , :].shape)
                dec_input = torch.cat((bos_token,emb_target[:, first_element , :],emb_target[:,1:,:][:, filtered_perm , :]), dim=1)
                print(dec_input.shape)
                # dec_input = torch.cat((bos_token, emb_target[:,current_perm,:]), dim=1)

            if use_pos_emb:
                # Add position embedding if they exist.
                dec_input = dec_input + self.pos_embed.to(emb_target.dtype)

            # dec_input has the same shape as emb_target, which is [B, N, D]
            print(dec_input.shape)
            dec_input = self.input_proj(dec_input)
            print(dec_input.shape)
    
            # Apply the decoder
            print(slots.shape)
            dec_input_slots = self.slot_proj(slots) # shape: [B, num_slots, D]
            print(dec_input_slots.shape)

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
        emb_input, token_emb, token_indices = self.forward_encoder(image, self.encoder)
        print(emb_input.shape)

        with torch.no_grad():
            if self.second_encoder is not None:
                emb_target,_,_ = self.forward_encoder(image, self.second_encoder)
            else:
                if self.use_token_embs:
                    emb_target = token_emb.clone().detach()
                else:
                    emb_target = emb_input.clone().detach()
        # emb_target shape: B, N, D
        # print(emb_target.shape)

        # Apply the slot attention
        slots, slots_attns, init_slots, attn_logits = self.slot_attn(emb_input)
        print("Slots shapes are:")
        print(slots.shape)
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

            loss_out = ((emb_target[:,1:,:] - dec_recon) ** 2).sum()/(B*H_enc*W_enc*self.d_model)# changed emb_target shape

        # Reshape the slot and decoder-slot attentions.
        # slots_attns = slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)
        slots_attns = slots_attns[:,1:,:].transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

        dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

        


        return loss_out, slots_attns, dec_slots_attns, slots, dec_recon, attn_logits
