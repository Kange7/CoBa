from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import helper_tf_util
import time


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)




    def get_loss(self, logits, labels, pre_cal_weights, dice_weight=0.1):
        # 
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        ce_loss = tf.reduce_mean(weighted_losses)
    
        # Dice Loss
        probs = tf.nn.softmax(logits, axis=-1)
        w = 1. / (tf.reduce_sum(one_hot_labels, axis=0) ** 2 + 1e-8)  # [C]
        intersection = w * tf.reduce_sum(probs * one_hot_labels, axis=0)  # [C]
        union = w * tf.reduce_sum(probs + one_hot_labels, axis=0) + 1e-8  # [C]
        dice_loss = 1.0 - 2. * tf.reduce_sum(intersection) / tf.reduce_sum(union)
    
        return ce_loss + dice_weight * dice_loss


    '''
    def cross_subcloud_attention(self, feat_low, feat_high, xyz_low, xyz_high, name, is_training):

        with tf.variable_scope(name):
            # 
            feat_low = tf.squeeze(feat_low, axis=2)  # [B, N1, d]
            feat_high = tf.squeeze(feat_high, axis=2)  # [B, N2, d]

            # 
            delta_xyz = tf.expand_dims(xyz_low, axis=2) - tf.expand_dims(xyz_high, axis=1)  # [B, N1, N2, 3]
            # delta_dist = tf.sqrt(tf.reduce_sum(tf.square(delta_xyz), axis=-1))  # [B, N1, N2]
            delta_dist = tf.sqrt(tf.reduce_sum(tf.square(delta_xyz), axis=-1, keepdims=True))  # [B, N1, N2, 1] (Rank=4)
            pos_enc = tf.concat([delta_dist, delta_xyz], axis=-1)  # [B, N1, N2, 4]
            pos_enc = tf.layers.dense(pos_enc, units=feat_low.shape[-1], name='pos_enc_mlp')  # [B, N1, N2, d]
            # pos_enc = tf.layers.dense(pos_enc, units=feat_high.shape[-1], name='pos_enc_mlp')  # [B, N1, N2, d]
            

            #
            q = tf.layers.dense(feat_low, feat_low.shape[-1], name='q_proj')  # [B, N1, d]
            k = tf.layers.dense(feat_high, feat_high.shape[-1], name='k_proj')  # [B, N2, d]
            v = tf.layers.dense(feat_high, feat_high.shape[-1], name='v_proj')  # [B, N2, d]

            # 
            # k = tf.expand_dims(k, axis=1) + pos_enc  # [B, N1, N2, d]
            k = tf.expand_dims(k, axis=1)
            k = tf.reduce_mean(k, axis=2)  # [B, N1, d]
            d=k.shape[-1].value

            # 
            attn_logits = tf.matmul(q, k, transpose_b=True)  # [B, N1, N1]
            attn_weights = tf.nn.softmax(attn_logits / tf.sqrt(tf.cast(d, tf.float32)), axis=-1)
            # print("attn_weights.shape:", attn_weights.shape)
            
            # 
            v_expanded = tf.expand_dims(v, axis=1)  # [B, 1, N2, d]
            # print("v_expanded.shape:", v_expanded.shape)
            v = tf.tile(v_expanded, [1, tf.shape(q)[1], 1, 1])  # [B, N1, N2, d]
            # print("v.shape:", v.shape)
            # attended_feat = tf.matmul(attn_weights, v)  # [B, N1, d]
            # attended_feat = tf.matmul(v, attn_weights)  # [B, N1, d]
            attended_feat = tf.reduce_sum(attn_weights * v, axis=2)  # [B, N1, d]
            # print("attended_feat.shape:", attended_feat.shape)

            # 
            attended_feat = tf.layers.dense(attended_feat, feat_low.shape[-1], name='out_proj')
            output = feat_low + attended_feat  # [B, N1, d]
            # print("output.shape:", output.shape)
            # print()
            return tf.expand_dims(output, axis=2)  # [B, N1, 1, d]
    '''

    '''
    def cross_subcloud_attention(self, feat_low, feat_high, xyz_low, xyz_high, name, is_training):
        """
        """
        with tf.variable_scope(name):
            # 
            feat_low = tf.squeeze(feat_low, axis=2)  # [B, N1, d]
            feat_high = tf.squeeze(feat_high, axis=2)  # [B, N2, d]

            # -
            # delta_xyz = tf.expand_dims(xyz_low, axis=2) - tf.expand_dims(xyz_high, axis=1)  # [B, N1, N2, 3]
            # delta_dist = tf.sqrt(tf.reduce_sum(tf.square(delta_xyz), axis=-1, keepdims=True))  # [B, N1, N2, 1]
            # pos_enc = tf.concat([delta_dist, delta_xyz], axis=-1)  # [B, N1, N2, 4]
            # pos_bias = tf.layers.dense(pos_enc, units=1, name='pos_bias_mlp')  # [B, N1, N2, 1]
            # pos_bias = tf.squeeze(pos_bias, axis=-1)  # [B, N1, N2]
            # print("pos_bias.shape:", pos_bias.shape)

            # --- --
            projection_dim = 64
            q = tf.layers.dense(feat_low, projection_dim, name='q_proj')  # [B, N1, 64]
            k = tf.layers.dense(feat_high, projection_dim, name='k_proj')   # [B, N2, 64]
            v = tf.layers.dense(feat_high, projection_dim, name='v_proj')   # [B, N2, 64]

            # --- ---
            attn_logits = tf.matmul(q, k, transpose_b=True)  # [B, N1, N2]
            # print("attn_logits.shape:", attn_logits.shape)
            # attn_logits += pos_bias  #
            attn_logits = attn_logits / tf.sqrt(tf.cast(projection_dim, tf.float32))
            # print("attn_logits.shape:", attn_logits.shape)
            # attn_logits += pos_bias  #
            attn_weights = tf.nn.softmax(attn_logits, axis=-1)  # [B, N1, N2]

            # --- -
            attended_feat = tf.matmul(attn_weights, v)  # [B, N1, 64]

            # --- ---
            attended_feat = tf.layers.dense(attended_feat, feat_low.shape[-1], name='out_proj')
            output = feat_low + attended_feat  # [B, N1, d]
            return tf.expand_dims(output, axis=2)  # [B, N1, 1, d]
    '''

    ''' 
    def mult_heat_att_glue(self, feature, d_out, name, num_head, neigh, is_training, xyz, nn):

        d_out_nei = d_out
        # feature = self.mlp(feature, d_out_nei, name + 'star', is_training)
        num = tf.shape(xyz)[2]

        xyz = tf.tile(tf.expand_dims(xyz[:, :, 0, :], axis=2), [1, 1, num, 1]) - xyz  # 
        xyz = tf.layers.dense(xyz, d_out_nei, activation=None, name=name + 'xyz0')
        xyz = self.GLUE(tf.layers.batch_normalization(xyz, -1, 0.99, 1e-6, training=is_training))

        # feature = tf.layers.dense(feature, d_out_nei, activation=None, name=name + 'fea0')
        # feature = self.GLUE(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = self.mlp(feature, d_out_nei, name + 'fea0', is_training)

        v_i = self.random_gather(feature, neigh)  # 
        q_i = self.random_gather(feature, neigh)  # 
        k_i = self.random_gather(feature, neigh)
        q_i = tf.concat([q_i, xyz], axis=-1)
        k_i = tf.concat([q_i, xyz], axis=-1)
        attn_i = q_i - k_i
        attn_i = tf.layers.dense(attn_i, d_out, activation=None, name=name + 'end')
        v_i = tf.concat([v_i, xyz], axis=-1)
        q_i = tf.layers.dense(q_i, d_out_nei, activation=None, name=name + 'fff')  # (B, N, 32)
        v_i = tf.layers.dense(v_i, d_out_nei, activation=None, name=name + 'v0o')

        # q_i = q_i + xyz
        q_i = tf.nn.softmax(q_i, axis=-2)
        z_i = tf.reduce_sum(q_i * v_i, axis=2)

        z_i = tf.layers.dense(z_i, d_out, activation=None, name=name + 'end')
        z_i = self.GLUE(tf.layers.batch_normalization(z_i, -1, 0.99, 1e-6, training=is_training))

        return z_i
    '''  

    '''
    def Transformer_block(self, feature, neigh_idx, d_model, d_out, name, is_training):
        # d_in = feature.get_shape()[-1].value
        d_in = feature
        # print(d_in)
        # f_xyz = self.relative_pos_encoding_v2(xyz, neigh_idx)
        # f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        # 
        q = tf.layers.dense(d_in, d_model, name=name+'_q')  # [B, N, 1, d_model]
        # q = helper_tf_util.conv2d(d_in, d_model, [1, 1], name + 'mlp_q', [1, 1], 'VALID', True, is_training)
        # print("q.shape:", q.shape)
        k = tf.layers.dense(d_in, d_model, name=name+'_k')    # [B, N, K, d_model]
        # k = helper_tf_util.conv2d(d_in, d_model, [1, 1], name + 'mlp_k', [1, 1], 'VALID', True, is_training)
        # print("k.shape:", k.shape)
        v = tf.layers.dense(d_in, d_model, name=name+'_v')  # [B, N, K, d_model]
        # v = helper_tf_util.conv2d(d_in, d_model, [1, 1], name + 'mlp_v', [1, 1], 'VALID', True, is_training)
        # print("v.shape:", v.shape)
        f = tf.layers.dense(d_in, d_out, name=name+'_f')
        # f = helper_tf_util.conv2d(d_in, d_out, [1, 1], name + 'mlp_f', [1, 1], 'VALID', True, is_training)
        # print("k.shape:", k.shape)
        d_k = k.shape[-1]
        # 
        attn_logits = tf.matmul(q, k, transpose_b=True)  # [B, N, h, 1, K]
        # attn_logits = q - k
        attn_logits = attn_logits / tf.sqrt(tf.cast(d_k, tf.float32))
        
        #
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        
        # 
        attn_output = tf.matmul(attn_weights, v)  # [B, N, h, 1, d_k]
        
        # 
        # attn_output = self.combine_heads(attn_output)  # [B, N, 1, d_model]
        
        # 
        output = tf.layers.dense(attn_output, d_out, name=name+'_proj')
        output = tf.keras.layers.LayerNormalization()(output + f)
        return output
    '''
    

    '''
    def relative_pos_encoding_v2(self, xyz, neigh_idx):
    #  [B, N, K, 3]
    neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
    
    # 
    delta_xyz = xyz[:, :, None, :] - neighbor_xyz  # [B, N, K, 3]
    delta_dist = tf.sqrt(tf.reduce_sum(delta_xyz**2, axis=-1, keepdims=True))
    
    # 
    freq = tf.range(0, 64, dtype=tf.float32)  # 
    sin_enc = tf.sin(delta_dist * (10 ** (freq[None,None,None,:] / 64)))
    cos_enc = tf.cos(delta_dist * (10 ** (freq[None,None,None,:] / 64)))
    
    return tf.concat([delta_xyz, sin_enc, cos_enc], axis=-1)  # [B, N, K, 3+128]
    '''

    '''
    def relative_pos_encoding_v2(self, xyz, neigh_idx, sigma=1.0):
        # [B, N, K, 3]
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        
        # 
        delta_xyz = xyz[:, :, None, :] - neighbor_xyz  # [B, N, K, 3]
        
        # 
        delta_x = delta_xyz[..., 0:1]  # [B, N, K, 1]
        delta_y = delta_xyz[..., 1:2]
        delta_z = delta_xyz[..., 2:3]
        
        # 
        gamma_x = tf.exp(-tf.square(delta_x) / (2 * sigma**2))
        gamma_y = tf.exp(-tf.square(delta_y) / (2 * sigma**2))
        gamma_z = tf.exp(-tf.square(delta_z) / (2 * sigma**2))
        
        # 
        delta_dist = tf.sqrt(tf.reduce_sum(delta_xyz**2, axis=-1, keepdims=True))
        return tf.concat([delta_xyz, gamma_x, gamma_y, gamma_z, delta_dist], axis=-1)
    '''
    def relative_pos_encoding_v2(self, xyz, neigh_idx, sigma=0.35):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        # print("relative_xyz.shape:", relative_xyz.shape)
        # 
        delta_x = relative_xyz[..., 0:1]  # [B, N, K, 1]
        delta_y = relative_xyz[..., 1:2]
        delta_z = relative_xyz[..., 2:3]
        
        # 
        gamma_x = tf.exp(-tf.square(delta_x) / (2 * sigma**2))
        gamma_y = tf.exp(-tf.square(delta_y) / (2 * sigma**2))
        gamma_z = tf.exp(-tf.square(delta_z) / (2 * sigma**2))
        gamma_xyz = tf.concat([gamma_x, gamma_y, gamma_z], axis=-1)
        # gamma_xyz = tf.exp(-tf.square(relative_xyz) / (2 * sigma**2))
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        # relative_dis = tf.sqrt(tf.reduce_sum(tf.square(gamma_xyz), axis=-1, keepdims=True))
        # relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz, gamma_xyz], axis=-1)  # 
        relative_feature = tf.concat([relative_dis, xyz_tile, neighbor_xyz, gamma_xyz], axis=-1)
        # relative_feature = tf.concat([relative_dis, gamma_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features
