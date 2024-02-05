import torch
import torch.nn as nn
from models.loss import batch_episym
from models.vanilla_transformer import TransformerLayer
import torch.nn.functional as F
import models.resnet34 as resnet


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        # e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')  # pytorch 2.0
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    if logits.shape[1] == 2:
        mask = logits[:, 0, :, 0]
        weights = logits[:, 1, :, 0]

        mask = torch.sigmoid(mask)
        weights = torch.exp(weights) * mask
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
    elif logits.shape[1] == 1:
        weights = torch.relu(torch.tanh(logits))  # tanh and relu

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.GELU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return F.gelu(out)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]

    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class GAB(nn.Module):
    def __init__(self, in_channels=128, k=9):
        super(GAB, self).__init__()
        self.knn_num = k
        self.reduction = 3

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=True),
            nn.InstanceNorm2d(in_channels, eps=1e-5),
            nn.BatchNorm2d(in_channels)
        )

        self.pointcn1 = nn.Sequential(
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        self.pointcn2 = nn.Sequential(
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        if self.knn_num == 9:
            self.spatial_att = nn.Sequential(
                nn.Conv2d(self.knn_num, self.reduction, kernel_size=1),
                nn.BatchNorm2d(self.reduction), nn.GELU(),
                nn.Conv2d(self.reduction, self.knn_num, kernel_size=1),
                nn.BatchNorm2d(self.knn_num),
            )
        if self.knn_num == 6:
            self.spatial_att = nn.Sequential(
                nn.Conv2d(self.knn_num, self.reduction, kernel_size=1),
                nn.BatchNorm2d(self.reduction), nn.GELU(),
                nn.Conv2d(self.reduction, self.knn_num, kernel_size=1),
                nn.BatchNorm2d(self.knn_num),
            )
        self.neighbor_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4), nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4), nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, knn_graph):
        knn_graph = self.embed(knn_graph)
        residual0 = knn_graph
        att1_0 = knn_graph.mean(dim=1).unsqueeze(dim=1)
        att2_0 = knn_graph.max(dim=1)[0].unsqueeze(dim=1)
        att0 = att1_0 + att2_0
        att0 = self.spatial_att(att0.transpose(1, 3))
        att0 = torch.sigmoid(att0).transpose(1, 3)
        knn_graph = knn_graph * att0
        knn_graph = knn_graph + residual0

        residual1 = knn_graph
        knn_graph = self.pointcn1(knn_graph)
        att1_1 = knn_graph.mean(dim=2).unsqueeze(dim=2)
        att2_1 = knn_graph.max(dim=2)[0].unsqueeze(dim=2)
        att1 = att1_1 + att2_1
        att1 = self.neighbor_att(att1)
        att1 = torch.sigmoid(att1)
        knn_graph = knn_graph * att1
        knn_graph = knn_graph + residual1

        residual2 = knn_graph
        knn_graph = self.pointcn2(knn_graph)
        att1_2 = knn_graph.mean(dim=3).unsqueeze(dim=3)
        att2_2 = knn_graph.max(dim=3)[0].unsqueeze(dim=3)
        att2 = att1_2 + att2_2
        att2 = self.channel_att(att2)
        att2 = torch.sigmoid(att2)
        knn_graph = knn_graph * att2
        knn_graph = knn_graph + residual2

        return knn_graph


class AnnularConv(nn.Module):
    def __init__(self, in_channels=128, k=9):
        super(AnnularConv, self).__init__()
        self.in_channel = in_channels
        self.knn_num = k

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )
        if self.knn_num == 6:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )

    def forward(self, features):
        B, C, N, _ = features.shape
        out = self.conv1(features)
        out = self.conv2(out)
        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(nn.InstanceNorm2d(in_channel, eps=1e-3),
                                  nn.BatchNorm2d(in_channel),
                                  nn.GELU(),
                                  nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class ContextFormer(nn.Module):
    def __init__(self, predict=False, out_channels=128, k_num=9, sampling_rate=0.5, num_heads=4, dropout=None, activation_fn='GELU'):
        super(ContextFormer, self).__init__()
        self.out_channels = out_channels
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.gab = GAB(self.out_channels, k=self.k_num)
        self.aggregator = AnnularConv(out_channels, k_num)

        self.encoder = nn.Sequential(
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False)
        )

        self.transformer = TransformerLayer(self.out_channels, num_heads, dropout=dropout, activation_fn=activation_fn)
        self.resnet_block = ResNet_Block(out_channels*2, out_channels, pre=True)

        self.linear_0 = nn.Conv2d(out_channels, 1, (1, 1))
        self.linear_1 = nn.Conv2d(out_channels, 1, (1, 1))

        if self.predict == True:
            self.inlier_predictor = nn.Sequential(
                ResNet_Block(self.out_channels, self.out_channels, pre=False),
                nn.Conv2d(self.out_channels, 2, (1, 1))
            )

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, embeddings, x, y):
        B, _, N, _ = x.size()
        src_keypts = x.squeeze(dim=1)[:, :, 0:2]
        tgt_keypts = x.squeeze(dim=1)[:, :, 2:4]
        with torch.no_grad():
            src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
            len_sim = src_dist - torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)

        out = get_graph_feature(embeddings, k=self.k_num)
        out = self.gab(out)
        out = self.aggregator(out)
        out = self.encoder(out)
        w0 = self.linear_0(out).view(B, -1)

        out_g = self.transformer(out.transpose(1, 2).squeeze(dim=-1), out.transpose(1, 2).squeeze(dim=-1), len_sim)[0]
        out_g = out_g.unsqueeze(dim=-1).transpose(1, 2)
        out = torch.cat([out, out_g], dim=1)
        out = self.resnet_block(out)
        w1 = self.linear_1(out).view(B, -1)

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N*self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N*self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            w2 = self.inlier_predictor(out)
            e_hat = weighted_8points(x_ds, w2)
            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


class Cross_Attention(nn.Module):
    def __init__(self, channels, head):
        super(Cross_Attention, self).__init__()
        self.head = head
        self.head_dim = channels // head

        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1)

        self.linear = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_linear = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, img1_tokens, img2_tokens, vs_flag=False):
        batch_size = img1_tokens.shape[0]

        query = self.q_proj(img1_tokens).view(batch_size, self.head, self.head_dim, -1)
        key = self.k_proj(img2_tokens).view(batch_size, self.head, self.head_dim, -1)
        value = self.v_proj(img2_tokens).view(batch_size, self.head, self.head_dim, -1)

        attention_scores = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim = -1)
        hidden_states = torch.einsum('bhnm,bhdm->bhdn', attention_scores, value).reshape(batch_size, self.head_dim * self.head, -1)
        hidden_states = self.linear(hidden_states)

        if vs_flag:
            return hidden_states
        else:
            output_states = img1_tokens + self.cat_linear(torch.cat([img1_tokens, hidden_states], dim=1))
            return output_states


class VCExtractor(nn.Module):
    def __init__(self, input_dim, internal_dim, output_dim):
        super(VCExtractor, self).__init__()
        head = 1

        self.cnn = resnet.resnet34(in_channels=3)
        self.cross_attention = Cross_Attention(input_dim, head)
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, internal_dim, kernel_size=1),
            nn.BatchNorm1d(internal_dim), nn.ReLU(),
            nn.Conv1d(internal_dim, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim), nn.ReLU(),
        )

    def forward(self, image1, image2):
        img1, img2 = self.cnn(image1.squeeze(dim=1)), self.cnn(image2.squeeze(dim=1))
        B, C, H, W = img1.shape
        img1 = img1.view(B, C, H * W)
        B, C, H, W = img2.shape
        img2 = img2.view(B, C, H * W)
        visual_cues = self.cross_attention(img1, img2)
        visual_cues = self.mlp(visual_cues)

        return visual_cues


class VSFusion_Light(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(VSFusion_Light, self).__init__()
        self.vs_flag = True
        self.cross_attention = Cross_Attention(hidden_dim, num_heads)

    def forward(self, visual_cues, spatial_cues):
        visual_spatial = self.cross_attention(spatial_cues.squeeze(dim=-1), visual_cues, self.vs_flag)
        return visual_spatial.unsqueeze(dim=-1)


class VSFormer(nn.Module):
    def __init__(self, config):
        super(VSFormer, self).__init__()
        out_channels = config.net_channels

        self.mlp0 = nn.Sequential(
            nn.Conv2d(4, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(6, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )
        self.resnet_blocks0 = nn.Sequential(
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False)
        )
        self.resnet_blocks1 = nn.Sequential(
            ResNet_Block(out_channels*2, out_channels, pre=True),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False)
        )

        self.vc_extractor = VCExtractor(input_dim=64, internal_dim=96, output_dim=out_channels)
        self.vsfusion = VSFusion_Light(hidden_dim=out_channels, num_heads=1)
        self.iter0 = ContextFormer(predict=False, out_channels=out_channels, k_num=9, sampling_rate=config.sr)
        self.iter1 = ContextFormer(predict=True, out_channels=out_channels, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y, img1, img2):
        B, _, N, _ = x.shape

        iteration_input0 = x.transpose(1, 3).contiguous()
        iteration_input0 = self.mlp0(iteration_input0)
        iteration_input0 = self.resnet_blocks0(iteration_input0)
        x1, y1, ws0, w_ds0 = self.iter0(iteration_input0, x, y)

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)

        # considering reducing the training cost, this paper only uses VSFusion in the second iteration
        spatial_cues = self.mlp1(x_.transpose(1, 3).contiguous())
        visual_cues = self.vc_extractor(img1, img2)
        visual_spatial = self.vsfusion(visual_cues, spatial_cues)
        visual_spatial = torch.cat([spatial_cues, visual_spatial], dim=1)
        iteration_input1 = self.resnet_blocks1(visual_spatial)
        x2, y2, ws1, w_ds1, e_hat = self.iter1(iteration_input1, x_, y1)

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat, [x, x, x1, x1, x2]