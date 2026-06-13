# S3 block - Tensorflow Implementation

import tensorflow as tf
from tensorflow.keras import layers


class S3Block(layers.Layer):
    """
    Self-Spatial Synergy (S3) block reconstructed to closely follow the diagram.

    Structure
    ---------
    1) Spatial Attention branch:
        Input -> Q, K, V projections
              -> Q @ K^T
              -> Softmax attention map
              -> attention-guided V
              -> Reshape
              -> Add with input

    2) Self Attention branch:
        Input -> MaxPool, AvgPool
              -> Concat
              -> Conv -> Sigmoid         (attention/gating map)
              -> gated feature refinement
              -> Conv -> SiLU

    3) Fusion:
        Concat(SpatialAttentionOutput, SelfAttentionOutput)
        -> Conv
        -> SiLU
    """

    def __init__(self, in_channels, inter_channels=None, **kwargs):
        super().__init__(**kwargs)

        if inter_channels is None:
            # Keeping C' = C is the strictest shape-compatible choice
            # with the residual addition shown in the diagram.
            inter_channels = in_channels

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # =====================================================
        # 1) Spatial Attention branch
        # =====================================================
        self.q_conv = layers.Conv2D(
            filters=inter_channels,
            kernel_size=1,
            use_bias=True,
            padding="valid"
        )
        self.k_conv = layers.Conv2D(
            filters=inter_channels,
            kernel_size=1,
            use_bias=True,
            padding="valid"
        )
        self.v_conv = layers.Conv2D(
            filters=inter_channels,
            kernel_size=1,
            use_bias=True,
            padding="valid"
        )

        # Optional restore projection if C' != C
        self.restore_conv = None
        if inter_channels != in_channels:
            self.restore_conv = layers.Conv2D(
                filters=in_channels,
                kernel_size=1,
                use_bias=True,
                padding="valid"
            )

        # =====================================================
        # 2) Self Attention branch
        # =====================================================
        self.max_pool = layers.MaxPooling2D(
            pool_size=3,
            strides=1,
            padding="same"
        )
        self.avg_pool = layers.AveragePooling2D(
            pool_size=3,
            strides=1,
            padding="same"
        )

        # From concatenated pooled features -> build attention gate
        self.self_gate_conv = layers.Conv2D(
            filters=in_channels,
            kernel_size=3,
            padding="same",
            use_bias=True
        )

        # Refine gated pooled features
        self.self_refine_conv = layers.Conv2D(
            filters=in_channels,
            kernel_size=3,
            padding="same",
            use_bias=True
        )

        # =====================================================
        # 3) Fusion
        # =====================================================
        self.fusion_conv = layers.Conv2D(
            filters=in_channels,
            kernel_size=3,
            padding="same",
            use_bias=True
        )

        self.sigmoid = layers.Activation("sigmoid")
        self.silu = tf.nn.silu

    def call(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            Shape: (B, H, W, C)

        Returns
        -------
        out : tf.Tensor
            Shape: (B, H, W, C)
        """
        input_shape = tf.shape(x)
        b = input_shape[0]
        h = input_shape[1]
        w = input_shape[2]

        # =====================================================
        # 1) Spatial Attention
        # =====================================================

        # Q: (B, H, W, C') -> (B, HW, C')
        q = self.q_conv(x)
        q = tf.reshape(q, [b, h * w, self.inter_channels])

        # K: (B, H, W, C') -> (B, C', HW)
        k = self.k_conv(x)
        k = tf.reshape(k, [b, h * w, self.inter_channels])
        k = tf.transpose(k, perm=[0, 2, 1])

        # V: (B, H, W, C') -> (B, C', HW)
        v = self.v_conv(x)
        v = tf.reshape(v, [b, h * w, self.inter_channels])
        v = tf.transpose(v, perm=[0, 2, 1])

        # Attention map: (B, HW, HW)
        attention_map = tf.matmul(q, k)
        attention_map = tf.nn.softmax(attention_map, axis=-1)

        # Apply attention to V: (B, C', HW)
        spatial_out = tf.matmul(v, tf.transpose(attention_map, perm=[0, 2, 1]))

        # Back to feature map: (B, H, W, C')
        spatial_out = tf.transpose(spatial_out, perm=[0, 2, 1])
        spatial_out = tf.reshape(spatial_out, [b, h, w, self.inter_channels])

        # Restore channel dimension if needed
        if self.restore_conv is not None:
            spatial_out = self.restore_conv(spatial_out)

        # Residual add exactly as shown in the top branch
        spatial_out = spatial_out + x

        # =====================================================
        # 2) Self Attention branch
        # =====================================================

        # Parallel pooling paths
        max_feat = self.max_pool(x)  # (B, H, W, C)
        avg_feat = self.avg_pool(x)  # (B, H, W, C)

        # Concat pooled outputs
        pooled_concat = tf.concat([max_feat, avg_feat], axis=-1)  # (B, H, W, 2C)

        # Gate path: Conv -> Sigmoid
        gate = self.self_gate_conv(pooled_concat)                 # (B, H, W, C)
        gate = self.sigmoid(gate)

        # To apply the gate on the concatenated pooled representation,
        # we expand the gate from (B, H, W, C) to (B, H, W, 2C)
        gate_expanded = tf.concat([gate, gate], axis=-1)         # (B, H, W, 2C)

        # Gated pooled representation
        gated_pooled = pooled_concat * gate_expanded             # (B, H, W, 2C)

        # Refinement path: Conv -> SiLU
        self_out = self.self_refine_conv(gated_pooled)           # (B, H, W, C)
        self_out = self.silu(self_out)

        # =====================================================
        # 3) Fusion
        # =====================================================

        # Concat outputs of the two branches
        fused = tf.concat([spatial_out, self_out], axis=-1)      # (B, H, W, 2C)

        # Final Conv -> SiLU
        out = self.fusion_conv(fused)                            # (B, H, W, C)
        out = self.silu(out)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "inter_channels": self.inter_channels,
        })
        return config




# # S3 block - PyTorch Implementation

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class S3Block(nn.Module):
#     """
#     Self-Spatial Synergy (S3) block reconstructed to closely follow the diagram.

#     Structure
#     ---------
#     1) Spatial Attention branch:
#         Input -> Q, K, V projections
#               -> Q @ K^T
#               -> Softmax attention map
#               -> attention-guided V
#               -> Reshape
#               -> Add with input

#     2) Self Attention branch:
#         Input -> MaxPool, AvgPool
#               -> Concat
#               -> Conv -> Sigmoid         (attention/gating map)
#               -> gated feature refinement
#               -> Conv -> SiLU

#     3) Fusion:
#         Concat(SpatialAttentionOutput, SelfAttentionOutput)
#         -> Conv
#         -> SiLU
#     """

#     def __init__(self, in_channels, inter_channels=None):
#         super().__init__()

#         if inter_channels is None:
#             # Keeping C' = C is the strictest shape-compatible choice
#             # with the residual addition shown in the diagram.
#             inter_channels = in_channels

#         self.in_channels = in_channels
#         self.inter_channels = inter_channels

#         # =====================================================
#         # 1) Spatial Attention branch
#         # =====================================================
#         self.q_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True)
#         self.k_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True)
#         self.v_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True)

#         # Optional restore projection if C' != C
#         self.restore_conv = None
#         if inter_channels != in_channels:
#             self.restore_conv = nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=True)

#         # =====================================================
#         # 2) Self Attention branch
#         # =====================================================
#         self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

#         # From concatenated pooled features -> build attention gate
#         self.self_gate_conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1, bias=True)

#         # Refine gated pooled features
#         self.self_refine_conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1, bias=True)

#         # =====================================================
#         # 3) Fusion
#         # =====================================================
#         self.fusion_conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1, bias=True)

#         self.sigmoid = nn.Sigmoid()
#         self.silu = nn.SiLU(inplace=True)

#     def forward(self, x):
#         """
#         Parameters
#         ----------
#         x : torch.Tensor
#             Shape: (B, C, H, W)

#         Returns
#         -------
#         out : torch.Tensor
#             Shape: (B, C, H, W)
#         """
#         b, c, h, w = x.shape

#         # =====================================================
#         # 1) Spatial Attention
#         # =====================================================

#         # Q: (B, C', H, W) -> (B, HW, C')
#         q = self.q_conv(x).view(b, self.inter_channels, h * w).permute(0, 2, 1)

#         # K: (B, C', H, W) -> (B, C', HW)
#         k = self.k_conv(x).view(b, self.inter_channels, h * w)

#         # V: (B, C', H, W) -> (B, C', HW)
#         v = self.v_conv(x).view(b, self.inter_channels, h * w)

#         # Attention map: (B, HW, HW)
#         attention_map = torch.bmm(q, k)
#         attention_map = F.softmax(attention_map, dim=-1)

#         # Apply attention to V: (B, C', HW)
#         spatial_out = torch.bmm(v, attention_map.transpose(1, 2))

#         # Back to feature map: (B, C', H, W)
#         spatial_out = spatial_out.view(b, self.inter_channels, h, w)

#         # Restore channel dimension if needed
#         if self.restore_conv is not None:
#             spatial_out = self.restore_conv(spatial_out)

#         # Residual add exactly as shown in the top branch
#         spatial_out = spatial_out + x

#         # =====================================================
#         # 2) Self Attention branch
#         # =====================================================

#         # Parallel pooling paths
#         max_feat = self.max_pool(x)  # (B, C, H, W)
#         avg_feat = self.avg_pool(x)  # (B, C, H, W)

#         # Concat pooled outputs
#         pooled_concat = torch.cat([max_feat, avg_feat], dim=1)  # (B, 2C, H, W)

#         # Gate path: Conv -> Sigmoid
#         gate = self.self_gate_conv(pooled_concat)               # (B, C, H, W)
#         gate = self.sigmoid(gate)

#         # To apply the gate on the concatenated pooled representation,
#         # we expand the gate from (B, C, H, W) to (B, 2C, H, W)
#         gate_expanded = torch.cat([gate, gate], dim=1)         # (B, 2C, H, W)

#         # Gated pooled representation
#         gated_pooled = pooled_concat * gate_expanded           # (B, 2C, H, W)

#         # Refinement path: Conv -> SiLU
#         self_out = self.self_refine_conv(gated_pooled)         # (B, C, H, W)
#         self_out = self.silu(self_out)

#         # =====================================================
#         # 3) Fusion
#         # =====================================================

#         # Concat outputs of the two branches
#         fused = torch.cat([spatial_out, self_out], dim=1)      # (B, 2C, H, W)

#         # Final Conv -> SiLU
#         out = self.fusion_conv(fused)                          # (B, C, H, W)
#         out = self.silu(out)

#         return out
