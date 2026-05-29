import torch
from einops import rearrange
import torch.nn.functional as F
from timm.layers import DropPath
import math


class PatchPartition(torch.nn.Module):
    """
    Patch Partition is the operation that is done at the beginning of the Swin transformer to create the patches on which the attention will be computed.
    It consists in concatenating the nearby patches in patches and reducing the dimensions with a linear transformation.
    """
    def __init__(self, patch_size, in_channels, out_channels):
        super(PatchPartition, self).__init__()
        self.patch_size = patch_size

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(patch_size, patch_size), stride=patch_size, bias=False)
        self.layer_norm = torch.nn.LayerNorm(out_channels, eps=1e-4)

    def forward(self, image):

        if len(image.shape) < 3:
            raise ValueError(f"The input must have at least 3 dimensions. Current : {image.shape}")

        _, _, H, W = image.shape

        # Adjust dimensions to be divisible by patch size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            added_pad_H = self.patch_size - (H % self.patch_size)
            added_pad_W = self.patch_size - (W % self.patch_size)
            image = F.pad(image, pad=(0, added_pad_W, 0, added_pad_H), mode='constant', value=0.0)
        x = self.conv2d(image)
        x = x.permute(0, 2, 3, 1).contiguous()  
        x = self.layer_norm(x)
        return x


class SwinRelativePositionEmbedding(torch.nn.Module):
    def __init__(self, window_size, n_head):
        super().__init__()

        """
        The aim is to create a map between the relative position of each point in the window and a unique index that will be used to look up the bias for that relative position in the relative_position_bias_table.
        """
        self.n_head = n_head

        # the number of relative positions that exist in the window
        num_positions = (2 * window_size - 1) ** 2

        self.relative_position_bias_table = torch.nn.Parameter(torch.randn(self.n_head, num_positions))

        # Create the index for rows and columns in the window.
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))
        coords_flat = coords.flatten(1)

        # compute the distance for each coordinate to each coordinate but still with indices row and column separated (distance dimension-wise).
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]

        # gather row and column distance from other point to other point and put them in the last dimension.
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # Range of indice will be from 0 to (2*Window_size - 1)^2-1 so we add the window size - 1 to make all indice positives.
        relative_coords += window_size - 1

        # Formula to make each relative position unique going from 2 dimensions to 1.
        relative_coords[:, :, 0] *= 2 * window_size - 1

        self.register_buffer("relative_position_index", relative_coords.sum(-1))


    def forward(self):
        return self.relative_position_bias_table[:, self.relative_position_index.view(-1)].view(
            (-1,) +self.relative_position_index.shape
        )



class SwinTransformerBlock(torch.nn.Module):
    def __init__(self, window_size, n_heads, query_size=32, mlp_factor=4, mask=None, dropout_rate=0.1, drop_path_rate=0.1):
        super(SwinTransformerBlock, self).__init__()

        self.in_channels = query_size*n_heads
        self.window_size = window_size
        self.query_size = query_size
        self.n_heads = n_heads
        if mask is not None:
            self.use_shift = True
            self.register_buffer("row_mask", mask[0])
            self.register_buffer("column_mask", mask[1])
        else:
            self.use_shift = False
            self.row_mask = None
            self.column_mask = None

        # Attention part
        self.query_proj = torch.nn.Linear(self.in_channels, self.in_channels)
        self.key_proj = torch.nn.Linear(self.in_channels, self.in_channels)
        self.value_proj = torch.nn.Linear(self.in_channels, self.in_channels)
        self.output_proj = torch.nn.Linear(self.in_channels, self.in_channels)
        self.logit_scale = torch.nn.Parameter(torch.ones(self.n_heads) * math.log(100.0))
        self.relative_pos_embedding = SwinRelativePositionEmbedding(self.window_size, self.n_heads)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.layer_norm_1 = torch.nn.LayerNorm(self.in_channels, eps=1e-4)

        # MLP part
        self.fc1 = torch.nn.Linear(self.in_channels, mlp_factor * self.in_channels)
        self.activation = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(mlp_factor * self.in_channels, self.in_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.layer_norm_2 = torch.nn.LayerNorm(self.in_channels, eps=1e-4)
        self.dropath = DropPath(drop_path_rate) if drop_path_rate > 0. else torch.nn.Identity()

        self.original_size = None

    def create_windows(self, input):
        """
        The aim of this function is to create the windows on which the attention will be computed and to create a mask for the padded values if the image size is not divisible by the window size.
         - If the image size is not divisible by the window size, we pad the image with zeros and create a mask with -1000.0 on the padded values to make sure they don't influence the attention computation (after applying softmax, they will be close to 0).
         - If we are in a shifted window transformer block, we roll the image and the mask to create the shifted windows.
         - Finally, we define the windows by unfolding the image and the mask.
         :param input: input feature map of shape (batch_size, height, width, channels)
         :return: windows of shape (batch_size, nb_h_windows, nb_w_windows, window_size*window_size, channels) and mask of shape (nb_h_windows, nb_w_windows, window_size*window_size)
        """

        _, H, W, _ = input.shape
        self.original_size = (H, W)

        mask_padding = torch.full_like(input[0, :, :, 0], 0, dtype=torch.float, device=input.device)

        # pad if necessary
        if H % self.window_size != 0 or W % self.window_size != 0:
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            input = F.pad(input, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0.0)
            mask_padding = F.pad(mask_padding, (0, pad_w, 0, pad_h), mode='constant', value=torch.tensor(float(-1000.0), dtype=torch.float))

        # roll if the window is shifted
        if self.use_shift:
            input = input.roll(shifts=[-(self.window_size // 2), -(self.window_size // 2)], dims=[1, 2])
            mask_padding = mask_padding.roll(shifts=[-(self.window_size // 2), -(self.window_size // 2)], dims=[0, 1])

        # define the windows
        x = input.unfold(1, self.window_size, self.window_size).unfold(2, self.window_size, self.window_size)
        mask_padding = mask_padding.unfold(0, self.window_size, self.window_size).unfold(1, self.window_size, self.window_size)
        return  x.permute(0, 1, 2, 4, 5, 3).contiguous(), mask_padding.contiguous()

    def remove_windows(self, input):
        # Input shape : B, nb_h_windows, nb_w_windows, self.window_size*self.window_size, self.in_channels
        B, nb_h_windows, nb_w_windows, _, C = input.shape
        H, W = self.original_size

        # free windows
        x = rearrange(input, 'B n_h n_w (w_1 w_2) C ->B (n_h w_1) (n_w w_2) C',B=B, w_1=self.window_size, w_2=self.window_size, n_h=nb_h_windows, n_w=nb_w_windows, C=C)

        # roll back
        if self.use_shift:
            x = x.roll(shifts=[self.window_size // 2, self.window_size // 2], dims=[1, 2])

        # truncate the added padding
        x = x[:, :H, :W]

        return x

    def forward(self, input):

        # pad, create a mask on the padded values and create windows
        x, mask_padding = self.create_windows(input)

        batch_size, nb_h_windows, nb_w_windows = x.shape[0:3]

        # generate keys, queries, values
        keys = self.key_proj(x)
        queries = self.query_proj(x)
        values = self.value_proj(x)

        # prepare for the attention computation
        keys = keys.view(batch_size, nb_h_windows, nb_w_windows, self.window_size*self.window_size, self.n_heads, self.query_size)
        queries = queries.view(batch_size, nb_h_windows, nb_w_windows, self.window_size * self.window_size, self.n_heads, self.query_size)
        values = values.view(batch_size, nb_h_windows, nb_w_windows, self.window_size * self.window_size, self.n_heads, self.query_size)

        # normalize Keys and Queries to compute cosine similarity | also
        keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-6)
        queries = queries / (queries.norm(dim=-1, keepdim=True) + 1e-6)

        scale = self.logit_scale.exp().clamp(max=100.0)
        x = torch.einsum("...whc,...jhc -> ...hwj", queries, keys)

        x = x * scale.view(1,1,1,self.n_heads,1,1)
        x = x +  self.relative_pos_embedding()

        # if we are in a shifted window transformer block => mask the unrelated values
        if self.use_shift:
            x[:, -1, :] += self.row_mask
            x[:, :, -1] += self.column_mask

        # mask the padded values
        mask_padding = mask_padding.view(nb_h_windows,nb_w_windows, self.window_size * self.window_size)
        mask_padding = mask_padding[:,:,None,None,:].expand(nb_h_windows,nb_w_windows, self.n_heads, self.window_size * self.window_size, self.window_size * self.window_size)

        x += mask_padding

        # finish the attention mecanism
        x = torch.einsum("...hwj, ...jhc -> ...whc", self.softmax(x), values) # output shape : batch_size, nb_h_windows, nb_w_windows, self.window_size*self.window_size, self.n_heads, self.query_size
        x = x.contiguous().view(batch_size, nb_h_windows, nb_w_windows, self.window_size*self.window_size, self.n_heads * self.query_size)

        # Concatenate heads into the in_channels dimension 
        x = self.output_proj(x) # output shape : batch_size, nb_h_windows, nb_w_windows, self.window_size*self.window_size, self.in_channels

        # Free the windows
        x = self.remove_windows(x)

        # normalize according to the channel dimension
        x = self.layer_norm_1(x)

        # merge with the input
        x = self.dropath(x)  + input
        residual = x

        # MLP part
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # normalize according to the channel dimension
        x = self.layer_norm_2(x)

        # merge with the input
        x = self.dropath(x) + residual

        return x



class PatchMerging(torch.nn.Module):
    """
    Patch Merging is the operation that is done at the end of each stage of the Swin transformer (except for the last one) to reduce the spatial dimensions and increase the number of channels.
    It consists in merging together the features of each patch_merging_ratio*patch_merging_ratio adjacent patches and then applying a linear transformation to reduce the dimensions.
    """
    def __init__(self, patch_merging_ratio, in_channels):
        super(PatchMerging, self).__init__()

        self.in_channels = in_channels
        self.patch_merging_ratio = patch_merging_ratio
        self.layer_norm = torch.nn.LayerNorm(self.patch_merging_ratio * self.patch_merging_ratio * self.in_channels, eps=1e-4)
        self.reduce_dimensions = torch.nn.Linear(self.patch_merging_ratio * self.patch_merging_ratio * self.in_channels, self.patch_merging_ratio * self.in_channels, bias=False)

    def forward(self, input):
        B, H, W, C = input.shape

        # Adjust dimensions to be divisible by patch_merging_ratio
        if H % self.patch_merging_ratio != 0 or W % self.patch_merging_ratio != 0:
            added_pad_H = self.patch_merging_ratio - (H % self.patch_merging_ratio)
            added_pad_W = self.patch_merging_ratio - (W % self.patch_merging_ratio)
            image = F.pad(image, pad=(0, added_pad_W, 0, added_pad_H), mode='constant', value=0.0)

        x = input.unfold(1, self.patch_merging_ratio, self.patch_merging_ratio).unfold(2, self.patch_merging_ratio, self.patch_merging_ratio)
        x = x.permute(0, 1, 2, 4, 5, 3)
        x = x.reshape(B, H // self.patch_merging_ratio, W // self.patch_merging_ratio, self.patch_merging_ratio * self.patch_merging_ratio * self.in_channels)
        x = x.contiguous()
        x = self.layer_norm(x)
        return self.reduce_dimensions(x)



class SwinTransformer(torch.nn.Module):
    def __init__(self, patch_size=4, patch_merging_ratio=2, in_channels=3, layers=[2,2,6,2], query_size=32, n_heads=[3, 6, 12, 24], mlp_factor=4, window_size=7, dropout_rate=0.1, drop_path_rate=0.1):

        super(SwinTransformer, self).__init__()
        self.window_size = window_size


        self.patch_partition = PatchPartition(patch_size, in_channels, query_size*n_heads[0])

        self.masks = self.create_mask(self.window_size)

        self.stages = []
        swin_blocks = []
        for i, layer in enumerate(layers):

            for j in range(layer):
                if j%2!=0:
                    mask = self.masks
                else :
                    mask = None
                swin_blocks.append(SwinTransformerBlock(window_size=window_size, n_heads=n_heads[i], query_size=query_size, mlp_factor=mlp_factor, mask = mask, dropout_rate=dropout_rate, drop_path_rate=drop_path_rate))

            #add patch merging except for the last stage
            if i !=3 :
                swin_blocks.append(PatchMerging(patch_merging_ratio, n_heads[i]*query_size))

            self.stages.append(torch.nn.ModuleList(swin_blocks))
            swin_blocks = []

        self.stages = torch.nn.ModuleList(self.stages)

        self.final_layer_norm = torch.nn.LayerNorm(n_heads[-1] * query_size, eps=1e-4)

    
    # masking non-related tokens
    def create_mask(self, window_size):
        """
        Creates masks based on window size for the shifted window attention.
        This aim to mask the influence of non spacialy closed values that have been gathered by rolling values of the image (shifting window)
        :param window_size:
        :return: row_mask, column_mask
        row_mask : to apply on the last row of windows
        column_mask : to apply on the last column of windows
        """
        row_mask = torch.zeros((window_size ** 2, window_size ** 2))
        inf_value = torch.tensor(float('-inf'))

        row_mask[-window_size * (window_size // 2):, :-window_size * (window_size // 2)] = inf_value
        row_mask[:-window_size * (window_size // 2), -window_size * (window_size // 2):] = inf_value
        column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=window_size, w2=window_size)
        return row_mask, column_mask


    def forward(self, input):

        x = self.patch_partition(input)

        for i, stage in enumerate(self.stages):

            if i!=len(self.stages)-1:
                for block in stage[:-1]:
                    x = block(x)


                #patch merging
                x = stage[-1](x)

            #last block doesn't do patches merging
            else :
                for block in stage:
                    x = block(x)
        x = self.final_layer_norm(x)
        return x


if __name__ == "__main__":

    X = """
    batch_size = 1
    channels = 3
    h, w = 512, 512
    window_size = 7
    torch.set_printoptions(threshold=float('inf'))


    image_original = torch.arange(batch_size * channels * h * w).reshape(batch_size, channels, h, w).float()
    image_original = image_original/ image_original.norm()

    model = SwinTransformer()
    print(image_original.shape)
    for param in model.parameters():
        param.data.fill_(1.0)
    result = model(image_original)
    print(result[0].shape)
    """
