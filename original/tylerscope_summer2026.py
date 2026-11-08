"""Original summer-2026 research code, preserved for reference. NOT run by this repo.

This is the code the repo's ``# Original:`` comments quote. It is copied verbatim
(including original spellings and commented-out lines) from two notebooks:

  Part 1  llama_temp_vis.ipynb   imports + the TylerScope and TensorVisualizer classes
                                 (manual forward pass, attention "cameras", generation loop)
  Part 2  final_lamma_vis.ipynb  the cells that loaded the saved arrays and drew the
                                 COS / DICT / CONTROL comparison figures and GIF

The single-pass figures (max attention per head, average specialization, sliding
focus window) were made in a third notebook, llama_vis.ipynb, which is not reproduced here.

Where it went in the repo:
  TylerScope.manually_run_foward          -> attnscope/forward.py   layer_forward
  TylerScope.tokenize_input               -> attnscope/forward.py   embed
  TylerScope.static_cam_at_end_conv       -> attnscope/scope.py     end_window
  TylerScope.dynamic_camera_atten_slice   -> attnscope/scope.py     most_active_head + focus_window
  TylerScope.gen_from_prompt              -> attnscope/scope.py     generate_and_capture
  final_lamma_vis.ipynb cells 1-5         -> attnscope/plotting.py  load_slices ... make_gif
  TylerScope.construct_atten_plots, TensorVisualizer (live Qt viewer) -> not carried over

scripts/check_against_original.py runs the core functions from Part 1 against the
repo versions and checks the outputs are bit-for-bit identical.

Note: the model at the time was loaded as "mlx-community/Meta-Llama-3.1-8B-4bit" in the
saved notebook; the experiment's data was generated with Llama-3.2-3B-Instruct, which has
no lm_head, so gen_from_prompt's `self.model.lm_head(...)` line only works with the 8B.
"""

# =============================================================================
# Part 1: llama_temp_vis.ipynb
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
import mlx.core as mx
from mlx_lm import load
import mlx
import seaborn as sns
import time
import qtpy
import matplotlib
import os
import tqdm.notebook as tqdm
import textwrap
import numpy.ma as ma
import imageio






class TylerScope:


    def __init__(self,model_string):
        self.model,self.tokenizer = load(model_string)
        self.atten_out = []
        self.orig_atten_out = []
        self.tracked_heads = []
        self.mlp_out_mtwo = []
        self.mlp_out_mone = []
        self.text_gen = []
        
        
        

    def manually_run_foward(self,input, layer_num):

        curr_layer = self.model.layers[layer_num]
        num_atten_heads = self.model.args.num_attention_heads
        hidden_dim = self.model.args.hidden_size
        head_dim = hidden_dim // num_atten_heads
        swGelu = mlx.nn.SiLU()
        
        # We are going to the input is of size batch, seq_len, hidden_dim in this case.
        res = input
        
        # Next we norm the input.
        in_normed = curr_layer.input_layernorm(res)

        # Now we project and reshape.
        Q = curr_layer.self_attn.q_proj(in_normed)
        batch_len,seq_len,_ = Q.shape
        Q = Q.reshape(batch_len,seq_len,num_atten_heads,hidden_dim//num_atten_heads).transpose(0,2,1,3)

        V = curr_layer.self_attn.v_proj(in_normed)
        _,_,V_hidden_dim = V.shape
        V_num_heads = V_hidden_dim//head_dim
        V = V.reshape(batch_len,seq_len,V_num_heads,head_dim)
        G_V = num_atten_heads // V.shape[2]
        V = mx.expand_dims(V,axis=-2)
        V = mx.broadcast_to(V,(batch_len,seq_len,V_num_heads,G_V,head_dim))
        V = V.reshape(batch_len,seq_len,(V_num_heads*G_V),head_dim).transpose(0,2,1,3)
    

        K = curr_layer.self_attn.k_proj(in_normed)
        _,_,K_hidden_dim = K.shape
        K_num_heads = K_hidden_dim//head_dim
        K = K.reshape(batch_len,seq_len,K_num_heads,head_dim)
        # Now this is the number of head groups that we'll broadcast
        G_K = num_atten_heads // K.shape[2]
        # Now we actually do this expansion.
        K = mx.expand_dims(K,axis=-2)
        K = mx.broadcast_to(K,(batch_len,seq_len,K_num_heads,G_K,head_dim))
        K = K.reshape(batch_len,seq_len,(K_num_heads*G_K),head_dim).transpose(0,2,1,3)
        

        # Now we apply the rope positional system.
        Q = curr_layer.self_attn.rope(Q)
        K = curr_layer.self_attn.rope(K)

        #Next we do the QK^T operation to get the actual attention heads.
        unmasked_att = Q @ K.transpose(0,1,3,2)
        # Now we need to divide by the sqrt of the length of K
        unmasked_att = unmasked_att / mx.sqrt(head_dim)
        mask = mlx.nn.MultiHeadAttention.create_additive_causal_mask(unmasked_att.shape[-1])
        casual_att = mx.softmax((unmasked_att+mask),axis=-1)
        
        # Now we do the dot product with V.
        atten_out = casual_att @ V
        # Now we will get the heads back together.
        atten_out  = atten_out.transpose(0,2,1,3).reshape(batch_len,seq_len,hidden_dim)
        atten_out = curr_layer.self_attn.o_proj(atten_out)

        # Next we add to the residual stream 
        res = res + atten_out

        # Now we normalize attention out to get the final attention out.
        final_atten_out = curr_layer.post_attention_layernorm(res)
        
        # Now we are going to do the switch mlp layer.
        # First is the parralel projections.
        gt_proj = curr_layer.mlp.gate_proj(final_atten_out)
        up_proj = curr_layer.mlp.up_proj(final_atten_out)
        # Now we do the proper calculation.
        mlp_hidden = swGelu(gt_proj) * up_proj
        # Now we do the down proj layer.
        mlp_out = curr_layer.mlp.down_proj(mlp_hidden)

        # Now we add the result of the mlp to our residual stream.
        res = res + mlp_out

        # Now the final output is our residual stream and will be normed at the top of the next time this func is called.

        return res,casual_att
        
        



    def dynamic_camera_atten_slice(self,layer_att,orig_seq_len,tracking_head=None,kernel_size=3,cam_radius=30,track_option=0):
        # First we'll find the most intresting head to slice. I am making this optional
        # so that on the dict dump side I can just use the head that was previously found
        # to be the most intresting.
        if tracking_head is None:
            median_max_att = mlx.core.squeeze(mlx.core.median(mlx.core.max(layer_att,axis=-1),axis=-1))
            tracking_head = median_max_att.argmax(axis=-1)

        
        padding_sz = int(((kernel_size-1)//2))
        conv_lay = mlx.nn.AvgPool2d(kernel_size=(kernel_size,kernel_size),padding=(padding_sz),stride=1)


        
        good_head = mlx.core.expand_dims(layer_att[tracking_head,:,:],axis=(0,3))
        _,good_head_height, good_head_width,_ = good_head.shape
    


        center_flat_ind = 0

        if track_option == 0:
            # Now we are going to calculate where the camera should currently center on for each head independently.
            # NOTE you could also just slice before argmaxing to get the most intresting part of the head that is not part of the input prompt.
            center_flat_ind =  mlx.core.argmax(mlx.core.flatten(mlx.core.squeeze(conv_lay(good_head)),start_axis=0)).item()
        elif track_option == 1:
            mult = 1.75
            orig_t_mult = orig_seq_len*mult

            conv_map = mlx.core.squeeze(conv_lay(good_head))

            if good_head_height > orig_seq_len:
                conv_map = conv_map[(good_head_height-(good_head_height//4)):,(good_head_width-(good_head_width//4)):]
                #print(conv_map.shape)
                
            
            center_flat_ind =  mlx.core.argmax(mlx.core.flatten(conv_map,start_axis=0)).item()
        else:
            center_flat_ind = good_head_height * good_head_width # Assuming square seq_len,seq_len attention head

        # Now get rid of the one dimensions added for the conv layer
        good_head = mlx.core.squeeze(good_head)
    
        # Now we figure out where the camera is going to be centered.
        camera_row_cent = int(center_flat_ind // good_head_width)
        camera_col_cent = int(center_flat_ind % good_head_width)
        # Caclulate the actual amount we'll need to subtract for the padding.
        sub_amt = (cam_radius-1)//2

        # Next calc the clipped row and col slices
        row_min = np.clip((camera_row_cent-sub_amt),0,(good_head_height-1)).astype(int)
        row_max = np.clip((camera_row_cent+sub_amt),0,(good_head_height-1)).astype(int)
        col_min = np.clip((camera_col_cent-sub_amt),0,(good_head_width-1)).astype(int)
        col_max = np.clip((camera_col_cent+sub_amt),0,(good_head_width-1)).astype(int)

    

        # Now we'll actually slice the good and bad heads to get what the camera is currently seeing and we'll 
        # pad to make each of them the same size.
        good_sliced_mat = good_head[row_min:(row_max+1),col_min:(col_max+1)]
        
        # Need to calc top bottom left right padding here
        pad_top = np.abs(camera_row_cent-sub_amt) - row_min
        pad_bottom = (camera_row_cent+sub_amt) - row_max
        pad_left = np.abs(camera_col_cent-sub_amt) - col_min
        pad_right = (camera_col_cent+sub_amt) - col_max

        

        
        # Now actually make what we'll see
        good_lens_padded = mlx.core.pad(good_sliced_mat,pad_width=((pad_top,pad_bottom),(pad_left,pad_right)),constant_values=0)

        # Return the sliced and padded matrix plus the desired trackig head.
        return tracking_head, good_lens_padded
    
    def tokenize_input(self,prompt):
        tokenized_in = mx.expand_dims(mx.array(self.tokenizer.encode(prompt)),axis=0)

        embed_tokens = self.model.model.embed_tokens(tokenized_in)

        return embed_tokens
    
    def reset_trackers(self):
        self.atten_out = []
        self.mlp_out_mone = []
        self.mlp_out_mtwo = []
        self.tracked_heads = []
        self.text_gen = []
        self.orig_atten_out = []

    def static_cam_at_end_conv(self,layer_att,radius,track_head):
        
        
        layer_att = layer_att[0,track_head,:,:]
        seq_len,_ = layer_att.shape
        


        kernel_size = 3
        padding_sz = int(((kernel_size-1)//2))
        conv_lay = mlx.nn.AvgPool2d(kernel_size=(kernel_size,kernel_size),padding=(padding_sz),stride=1)
        conv_head = mlx.core.expand_dims(layer_att[:,:],axis=(0,3))

        
        conv_head = mx.squeeze(conv_lay(conv_head))[-radius:,-radius:]

        return track_head,conv_head

    
    def gen_from_prompt(self,promt_path,cam_radius,lay_guess_height,lay_guess_width,max_gen=50,attenCam=None,layerOutCamTwo=None,layerOutCamOne=None,tracking_head_op=0):
        # Clear the out lists.
        self.reset_trackers()

        prompt = ""

        eos_token_id = self.tokenizer.eos_token_id
        eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        with open(promt_path,"r") as f:
            prompt = " ".join(f.readlines())


        curr_gen = prompt


        gen_token_id = -1
        max_token_gens = max_gen
        num_tokens_gen = 0

        print("STARING GENERATION!!")

        prev_atten_out = None


        for _ in tqdm.tqdm(range(max_token_gens),total=max_token_gens):


            
            model_in = self.tokenize_input(curr_gen)
            prev_out = model_in

            atten_per_layer = []
            mlp_per_layer_mtwo = []
            mlp_per_layer_mone = []
            heads_per_layer = []
            orig_atten_out_per_layer = []

            for i in range(self.model.args.num_hidden_layers):
                
                next_out,atten_out = self.manually_run_foward(prev_out,i)
                #print(f"Layer num {i} atten out size was {atten_out.shape}")
                
                
                # Now in here is where we will look at the atten out for the current layer as 
                # well as view the current layers final out.
                #head_num,atten_slice = self.dynamic_camera_atten_slice(atten_out[0,:,:,:],orig_seq_len,cam_radius=cam_radius,track_option=tracking_head_op,tracking_head=17)
                head_num,atten_slice = self.static_cam_at_end_conv(atten_out,100,17)
                #print(f"Layer num {i} next_out size was {next_out.shape}")
                
                #if i == 27:
                    #print(mx.array_equal(atten_slice,prev_atten_out))


                next_out_secondlast = self.model.model.norm(next_out[0,-2,:])
                next_out_last = self.model.model.norm(next_out[0,-1,:])


                # Now append the current run to our trackers.
                atten_per_layer.append(atten_slice)
                heads_per_layer.append(head_num)
                orig_atten_out_per_layer.append(atten_out)

                mlp_per_layer_mtwo.append(next_out_secondlast)
                mlp_per_layer_mone.append(next_out_last)

                
                




                # Now we'll only visualize if the user wants to.
                shouldSleep = False
                if attenCam is not None:
                    attenCam.update(atten_slice,f"Tracking head {head_num+1} with cam R={cam_radius}")
                    shouldSleep = True

                if layerOutCamTwo is not None:    
                    next_out_secondlast = mx.softmax(next_out_secondlast,axis=-1)
                    next_out_secondlast = next_out_secondlast.reshape(lay_guess_height,lay_guess_width)
                    layerOutCamTwo.update(next_out_secondlast,f"For current out guess on tok pos =-2 is'")
                    shouldSleep = True

                if layerOutCamOne is not None:    
                    next_out_last = mx.softmax(next_out_last,axis=-1)
                    next_out_last = next_out_last.reshape(lay_guess_height,lay_guess_width)
                    layerOutCamOne.update(next_out_last,f"For current out guess on tok pos =-1 is'")
                    shouldSleep = True

                if shouldSleep:
                    time.sleep(5)


                

                prev_out = next_out
                prev_atten_out = atten_slice


            num_tokens_gen += 1

            
            atten_per_layer = mx.stack(atten_per_layer,axis=0)
            mlp_per_layer_mtwo = mx.stack(mlp_per_layer_mtwo,axis=0)
            mlp_per_layer_mone = mx.stack(mlp_per_layer_mone,axis=0)
            heads_per_layer = mx.array(heads_per_layer)
            orig_atten_out_per_layer = mx.stack(orig_atten_out_per_layer,axis=0)

            self.atten_out.append(atten_per_layer)
            self.mlp_out_mtwo.append(mlp_per_layer_mtwo)
            self.mlp_out_mone.append(mlp_per_layer_mone)
            self.tracked_heads.append(heads_per_layer)
            self.orig_atten_out.append(orig_atten_out_per_layer)

            

            gen_token_id = self.model.lm_head(mlp_per_layer_mone[-1,:]).argmax(axis=-1).item()
            curr_gen += self.tokenizer.decode(gen_token_id)
            
            self.text_gen.append(curr_gen)

            if gen_token_id == eos_token_id or gen_token_id == eot_token_id or num_tokens_gen == max_token_gens:
                break
            


        # Finally at the end of the gen we are going to stack all the tensors.
        self.atten_out = mx.stack(self.atten_out,axis=0)
        self.mlp_out_mone = mx.stack(self.mlp_out_mone,axis=0)
        self.mlp_out_mtwo = mx.stack(self.mlp_out_mtwo,axis=0)
        self.tracked_heads = mx.stack(self.tracked_heads,axis=0)
       

        return f"''{curr_gen}''"
    
    def construct_atten_plots(self, plts_path, super_title):
    # Prepare the timestamp once outside the loop for efficiency
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

        for gen_ind_curr in range(self.atten_out.shape[0]):
            for i in range(self.model.args.num_hidden_layers):
                fig, ax = plt.subplots(figsize=(8, 6))
                try:
                    head_num_curr = self.tracked_heads[i]

                    # 1. Create the image
                    img_obj = ax.imshow(self.atten_out[gen_ind_curr, i, :, :])
                    
                    # 2. Add the colorbar ONCE (and link it correctly)
                    fig.colorbar(img_obj, ax=ax)

                    # 3. Set the titles
                    fig.suptitle(f"{super_title} | gen {gen_ind_curr}", fontsize=16, fontweight='bold')
                    ax.set_title(f"Layer {i+1} | Head {head_num_curr}")

                    # 4. Add the subtitle (the generated text)
                    # We use a slightly lower y-coordinate (e.g., 0.85) to avoid the suptitle
                    wrapped_text = "\n".join(textwrap.wrap(self.text_gen[gen_ind_curr], width=50))
                    fig.text(0.5, 0.86, wrapped_text, ha='center', fontsize=9, style='italic')

                    # 5. Save with a valid string timestamp
                    filename = f"{plts_path}/{timestamp}_layer_{i}_head_{head_num_curr}_attentrack.png"
                    fig.savefig(filename, bbox_inches='tight') # bbox_inches='tight' prevents text cutoff

                    # REMOVED: plt.colorbar() <--- This was the error!

                except Exception as e:
                    print(f"Error saving layer {i}: {e}")
                finally:
                    # Ensures memory is always cleared
                    plt.close(fig)
   

    

    


class TensorVisualizer:
    def __init__(self, shape,super_title):
        self.shape = shape
        # Initialize the figure and the axis
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        
        # Create an initial empty image object
        # We use a dummy array of the correct shape
        self.img_obj = self.ax.imshow(np.zeros(shape), cmap=sns.color_palette("cubehelix", as_cmap=True), interpolation='nearest')
        
        # Add a colorbar so you can see the scale
        self.cbar = self.fig.colorbar(self.img_obj, ax=self.ax)
        
        self.fig.suptitle(super_title)
        # Turn on interactive mode (crucial!)
        plt.ion()
        plt.show()

    def update(self, new_tensor,title):
        """The magic method: updates the image data without re-drawing everything"""
        # 1. Update the data inside the existing image object
        new_tensor = np.array(new_tensor)

        self.img_obj.set_data(new_tensor)
        
        
        # 2. Adjust the color scaling if the values change range
        self.img_obj.set_clim(vmin=float(new_tensor.min()), vmax=float(new_tensor.max()))
        
        self.ax.set_title(title)
        # 3. Draw the changes and flush to the screen
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# =============================================================================
# Part 2: final_lamma_vis.ipynb (one notebook cell per section)
# =============================================================================

# ---- cell 0 ------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
import mlx.core as mx
from mlx_lm import load
import mlx
import seaborn as sns
import time
import qtpy
import matplotlib
import os
import tqdm
import textwrap
import numpy.ma as ma
import imageio

# ---- cell 1 ------------------------------------------------------------
atten_mlxs_pth = "llama_atten_slices/"
cos_atten = mx.load(f"{atten_mlxs_pth}/cos_atten_slice.npy")
dict_atten = mx.load(f"{atten_mlxs_pth}/dict_atten_slice.npy")
control_atten = mx.load(f"{atten_mlxs_pth}/control_atten_slice.npy")

cos_last = mx.load(f"{atten_mlxs_pth}/cos_lastmlp_slice.npy")
dict_last = mx.load(f"{atten_mlxs_pth}/dict_lastmlp_slice.npy")
control_last = mx.load(f"{atten_mlxs_pth}/control_lastmlp_slice.npy")

# ---- cell 2 ------------------------------------------------------------
atten_glob_vmin = min([mx.min(cos_atten,axis=(-4,-3,-2,-1)),mx.min(dict_atten,axis=(-4,-3,-2,-1)),mx.min(control_atten,axis=(-4,-3,-2,-1))])
atten_glob_vmax = max([mx.max(cos_atten,axis=(-4,-3,-2,-1)),mx.max(dict_atten,axis=(-4,-3,-2,-1)),mx.max(control_atten,axis=(-4,-3,-2,-1))])


mlp_glob_vmin = min([mx.min(mx.softmax(cos_last[-1,:,:],axis=-1),axis=(-2,-1)),mx.min(mx.softmax(dict_last[-1,:,:],axis=-1),axis=(-2,-1)),mx.min(mx.softmax(control_last[-1,:,:],axis=-1),axis=(-2,-1))])
mlp_glob_vmax = max([mx.max(mx.softmax(cos_last[-1,:,:],axis=-1),axis=(-2,-1)),mx.max(mx.softmax(dict_last[-1,:,:],axis=-1),axis=(-2,-1)),mx.max(mx.softmax(control_last[-1,:,:],axis=-1),axis=(-2,-1))])

# ---- cell 3 ------------------------------------------------------------
seq_len,num_layers,cam_height,cam_width = cos_atten.shape

# ---- cell 4 ------------------------------------------------------------
cam_height_with_pad = cam_height + 2
cam_width_with_pad = cam_width + 2

fig_paths = []


for layer_num in range(num_layers):

    # 12-Column Centered Layout
    layout = """
    .AAAABBBB..c
    ...CCCC....c
    DDDDDDDDDD.m
    EEEEEEEEEE.m
    FFFFFFFFFF.m
    """

    fig, ax_dict = plt.subplot_mosaic(
        layout, 
        figsize=(12, 18),          
        layout="constrained",      
        gridspec_kw={
            # 10 columns for plots, 1 for spacer, 1 for the colorbars
            "width_ratios": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.2, 0.3], 
            
            # Keep top rows tall for the squares, bottom rows short for the MLPs
            "height_ratios": [4, 4, 1.5, 1.5, 1.5] 
        }
    )

    border_val  = -1.0

    rows = cols = int(np.ceil(np.sqrt(seq_len)))
    canvas_cos = mx.full((cam_height_with_pad*rows,cam_width_with_pad*cols),vals=border_val,dtype=mx.float16)
    canvas_dict = mx.full((cam_height_with_pad*rows,cam_width_with_pad*cols),vals=border_val,dtype=mx.float16)
    canvas_control = mx.full((cam_height_with_pad*rows,cam_width_with_pad*cols),vals=border_val,dtype=mx.float16)

    # Now we'll loop through the seq len and add to the canvas.
    for idx in range(seq_len):


        # Now we need to calculate the current col num and row num.
        row_offset = (idx // rows)
        col_offset = (idx % cols)

        # NOw we get the index.
        row_ind = (cam_height_with_pad * row_offset)+1
        col_ind = (cam_width_with_pad * col_offset)+1


        
        canvas_cos[row_ind:(row_ind+cam_height),col_ind:(col_ind+cam_width)] = cos_atten[idx,layer_num,:,:]
        canvas_dict[row_ind:(row_ind+cam_height),col_ind:(col_ind+cam_width)] = dict_atten[idx,layer_num,:,:]
        canvas_control[row_ind:(row_ind+cam_height),col_ind:(col_ind+cam_width)] = control_atten[idx,layer_num,:,:]


    canvas_masked_cos = ma.masked_where(canvas_cos == -1.0, canvas_cos)
    vmin_cos = canvas_masked_cos.min()
    vmax_cos = canvas_masked_cos.max()

    canvas_masked_dict = ma.masked_where(canvas_dict == -1, canvas_dict)
    vmin_dict = canvas_masked_dict.min()
    vmax_dict = canvas_masked_dict.max()

    canvas_masked_control = ma.masked_where(canvas_control == -1, canvas_control)
    vmin_control = canvas_masked_control.min()
    vmax_control = canvas_masked_control.max()

    cmap = sns.cubehelix_palette(start=0.3, hue=0.9, rot=0.8, as_cmap=True)
    cmap.set_bad(color='lightgray')


    #plt.subplots(figsize=(10,10))
    atten_vis_cos = ax_dict["A"].imshow(canvas_masked_cos,cmap=cmap, interpolation='nearest', vmin=atten_glob_vmin, vmax=atten_glob_vmax)
    #fig.colorbar(atten_vis_cos)
    ax_dict["A"].set_xticks([])
    ax_dict["A"].set_yticks([])
    ax_dict["A"].set_title(f"COS ATTEN CONV 1:1")

    mlp_cos_curr = mx.softmax(cos_last[-1,layer_num,:])
    vmin_cos_mlp = mlp_cos_curr.min().item()
    vmax_cos_mlp = mlp_cos_curr.max().item()
    mlp_vs_cos = ax_dict["D"].imshow(mlp_cos_curr.reshape(24,128),cmap=sns.color_palette("mako",as_cmap=True),interpolation='nearest', vmin=mlp_glob_vmin, vmax=mlp_glob_vmax)
    #fig.colorbar(mlp_vs_cos)
    ax_dict["D"].set_title("COS MLP SOFTMAX")


    atten_vis_dict = ax_dict["B"].imshow(canvas_masked_dict,cmap=cmap, interpolation='nearest', vmin=atten_glob_vmin, vmax=atten_glob_vmax)
    #fig.colorbar(atten_vis_dict)
    ax_dict["B"].set_xticks([])
    ax_dict["B"].set_yticks([])
    ax_dict["B"].set_title(f"DICT ATTEN CONV 1:1")

    mlp_dict_curr = mx.softmax(dict_last[-1,layer_num,:])
    vmin_dict_mlp = mlp_dict_curr.min().item()
    vmax_dict_mlp = mlp_dict_curr.max().item()
    mlp_vs_dict = ax_dict["E"].imshow(mlp_dict_curr.reshape(24,128),cmap=sns.color_palette("mako",as_cmap=True),interpolation='nearest', vmin=mlp_glob_vmin, vmax=mlp_glob_vmax)
    #fig.colorbar(mlp_vs_dict)
    ax_dict["E"].set_title("DICT MLP SOFTMAX")

    atten_vis_control = ax_dict["C"].imshow(canvas_masked_control,cmap=cmap, interpolation='nearest', vmin=atten_glob_vmin, vmax=atten_glob_vmax)
    #fig.colorbar(atten_vis_control)
    ax_dict["C"].set_xticks([])
    ax_dict["C"].set_yticks([])
    ax_dict["C"].set_title(f"CONTROL ATTEN CONV 1:1")

    mlp_control_curr = mx.softmax(control_last[-1,layer_num,:])
    vmin_control_mlp = mlp_control_curr.min().item()
    vmax_control_mlp = mlp_control_curr.max().item()
    mlp_vs_control = ax_dict["F"].imshow(mlp_control_curr.reshape(24,128),cmap=sns.color_palette("mako",as_cmap=True),interpolation='nearest', vmin=mlp_glob_vmin, vmax=mlp_glob_vmax)
    #fig.colorbar(mlp_vs_control)
    ax_dict["F"].set_title("CONTROL MLP SOFTMAX")

    fig.suptitle(
    f"Gen tokens 0-{seq_len} head 17 & final token mlp; model layer {layer_num}",
    fontsize=18,          # You can use an integer or strings like 'x-large', 'xx-large'
    fontweight='bold'     # Makes the text bold
    #y=0.91
    )

    fig.colorbar(atten_vis_cos, cax=ax_dict['c'])
    fig.colorbar(mlp_vs_cos, cax=ax_dict['m'])

    file_path = f"final_vis/conv_scope_gen_len_h17_mlplast_lay{layer_num}.png"
    fig_paths.append(file_path)
    plt.savefig(file_path,bbox_inches="tight")
    plt.show()

# ---- cell 5 ------------------------------------------------------------
from PIL import Image
import imageio.v2 as imageio # v2 is safer for lists of images

# 1. Load all images using PIL
pil_images = [Image.open(img) for img in fig_paths]

# 2. Find the maximum width and height across all frames
max_width = max(img.width for img in pil_images)
max_height = max(img.height for img in pil_images)

gif_images = []

for img in pil_images:
    # 3. Create a new pure white image of the max dimensions
    background = Image.new('RGB', (max_width, max_height), (255, 255, 255))
    
    # 4. Paste the original image into the top-left corner (0,0) of the white canvas
    background.paste(img, (0, 0))
    
    # 5. Convert back to numpy array for imageio
    gif_images.append(background)

# 6. Save the perfectly uniform frames
imageio.mimsave(
    'final_vis/conv_scope_atten_gen_len_h17_mlplast.gif', 
    gif_images, 
    duration=1000, 
    loop=5
)
