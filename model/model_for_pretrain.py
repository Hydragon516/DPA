import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import math
import time

np.set_printoptions(suppress=True, threshold=1e5)


def resize(input, target_size=(352, 352)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))
        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        return out


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def get_prototype(x, ss_map):
    B, _, H, W = x.size()

    # ss_map: (B,128',H,W), spatially softmax-ed
    # x: (B,128,H,W)
    ss_map = ss_map.view(B, -1, H * W)
    x = x.view(B, -1, H * W)

    # prototype_block: (B,128',128)
    prototype_block = torch.bmm(ss_map, x.transpose(1, 2))
    return prototype_block


def get_correlation_map(x, prototype_block):
    B, C, H, W = x.size()

    # prototype_block: (B,N,C)
    # x: (B,C,H,W)
    # corr: (B,N,H,W), -1~1
    n_p = prototype_block / prototype_block.norm(dim=2, keepdim=True)
    n_x = x.view(B, C, -1) / x.view(B, C, -1).norm(dim=1, keepdim=True)
    corr = torch.bmm(n_p, n_x).view(B, -1, H, W)
    return corr


def get_ocr_vector(x):
    b, c, h, w = x.size()
    probs = x.view(b, c, -1)
    ss_map = F.softmax(probs, dim=2)
    ss_map = ss_map.view(b, c, h, w)
    pb = get_prototype(x, ss_map.clone().detach())
    return pb


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        layers = []
        in_channel = 3
        vgg_out_channels = (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M')
        for out_channel in vgg_out_channels:
            if out_channel == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = out_channel
        self.vgg = nn.ModuleList(layers)
        self.table = {'conv1_1': 0, 'conv1_2': 2, 'conv1_2_mp': 4,
                      'conv2_1': 5, 'conv2_2': 7, 'conv2_2_mp': 9,
                      'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_3_mp': 16,
                      'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21, 'conv4_3_mp': 23,
                      'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28, 'conv5_3_mp': 30, 'final': 31}

    def forward(self, feats, start_layer_name, end_layer_name):
        start_idx = self.table[start_layer_name]
        end_idx = self.table[end_layer_name]
        for idx in range(start_idx, end_idx):
            feats = self.vgg[idx](feats)
        return feats


class Prediction(nn.Module):
    def __init__(self, in_channel):
        super(Prediction, self).__init__()
        self.pred = nn.Sequential(nn.Conv2d(in_channel, 1, 1), nn.Sigmoid())

    def forward(self, feats):
        pred = self.pred(feats)
        return pred


class Res(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1),
                                   nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channel, in_channel, 3, 1, 1)
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                                   nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True)
                                   )

    def forward(self, feats):
        feats = feats + self.conv1(feats)
        feats = F.relu(feats, inplace=True)
        feats = self.conv2(feats)
        return feats


class Decoder_Block(nn.Module):
    def __init__(self, in_channel):
        super(Decoder_Block, self).__init__()
        self.cmprs = nn.Conv2d(in_channel, 64, 1)
        self.merge_conv = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                                        nn.Conv2d(96, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pred = Prediction(32)

    def forward(self, low_level_feats, cosal_map, old_feats):
        _, _, H, W = low_level_feats.shape
        cosal_map = resize(cosal_map, [H, W])
        old_feats = resize(old_feats, [H, W])
        cmprs = self.cmprs(low_level_feats)
        new_feats = self.merge_conv(torch.cat([cmprs * cosal_map, old_feats], dim=1))
        new_cosal_map = self.pred(new_feats)
        return new_feats, new_cosal_map


class Transformer(nn.Module):
    def __init__(self, in_channels):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.theta = nn.Linear(self.in_channels, self.inter_channels)
        self.phi = nn.Linear(self.in_channels, self.inter_channels)
        self.g = nn.Linear(self.in_channels, self.inter_channels)
        self.W = nn.Linear(self.inter_channels, self.in_channels)

    def forward(self, ori_feature):
        ori_feature = ori_feature.permute(0, 2, 1)
        feature = self.bn_relu(ori_feature)
        feature = feature.permute(0, 2, 1)
        B, N, C = feature.size()

        x_theta = self.theta(feature)
        x_phi = self.phi(feature)
        x_phi = x_phi.permute(0, 2, 1)
        attention = torch.matmul(x_theta, x_phi)

        f_div_C = F.softmax(attention, dim=-1)
        g_x = self.g(feature)
        y = torch.matmul(f_div_C, g_x)
        W_y = self.W(y).contiguous().view(B, N, C)
        att_fea = ori_feature.permute(0, 2, 1) + W_y
        return att_fea


class Graph_Attention_Network(nn.Module):
    def __init__(self, in_channels):
        super(Graph_Attention_Network, self).__init__()
        self.TF = Transformer(in_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, prototype_block):
        att_prototype_block = self.TF(prototype_block)
        prototype_for_graph = att_prototype_block.permute(0, 2, 1)
        graph_prototype = get_graph_feature(prototype_for_graph, k=10)
        graph_prototype = self.conv1(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(graph_prototype, k=10)
        graph_prototype = self.conv2(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(graph_prototype, k=10)
        graph_prototype = self.conv3(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]
        graph_prototype_block = graph_prototype.permute(0, 2, 1)
        return graph_prototype_block


class Prototype_Correlation_Generation(nn.Module):
    def __init__(self, in_channels):
        super(Prototype_Correlation_Generation, self).__init__()
        self.GAN = Graph_Attention_Network(in_channels)
        self.out = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1),
                                 nn.BatchNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 )

    def forward(self, x):
        pb = get_ocr_vector(x)
        graph_pb = self.GAN(pb)
        cm = get_correlation_map(x, graph_pb)
        return cm


class Domain_Alignment_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Domain_Alignment_Module, self).__init__()
        self.rgb_OCR = Prototype_Correlation_Generation(hidden_channels)
        self.rgb_fusion = nn.Sequential(nn.Conv2d(in_channels + hidden_channels, in_channels, 1),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU(inplace=True),
                                       )
    
    def forward(self, rgb, rgb_sq):
        rgb_da = self.rgb_OCR(rgb_sq)
        rgb = self.rgb_fusion(torch.cat([rgb, rgb_da], dim=1))
        return rgb


class DATA(nn.Module):
    def __init__(self):
        super(DATA, self).__init__()
        self.rgb_encoder = VGG16()

        self.rgb_sq4 = nn.Sequential(nn.Conv2d(512, 128, 1))
        self.rgb_sq5 = nn.Sequential(nn.Conv2d(512, 128, 1))

        self.rgb_COMP4 = nn.Sequential(nn.Conv2d(512, 128, 1))
        self.rgb_COMP5 = nn.Sequential(nn.Conv2d(512, 128, 1))
        self.rgb_aspp = ASPP(128, 128)
        self.rgb_COMP6 = nn.Sequential(nn.MaxPool2d(2, 2),
                                       nn.Conv2d(512, 128, 1),
                                       nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
                                       )

        self.DAM4 = Domain_Alignment_Module(512, 128)
        self.DAM5 = Domain_Alignment_Module(512, 128)

        self.rgb_merge_co_56 = Res(128 * 2, 128)
        self.rgb_merge_co_45 = nn.Sequential(Res(128 * 2, 128), nn.Conv2d(128, 32, 1))
        self.rgb_get_pred_4 = Prediction(32)
        self.rgb_refine_3 = Decoder_Block(256)
        self.rgb_refine_2 = Decoder_Block(128)
        self.rgb_refine_1 = Decoder_Block(64)
        

    def forward(self, image, s):
        ### STAGE 1 ###
        rgb_f1 = self.rgb_encoder(image, 'conv1_1', 'conv1_2_mp')
        rgb_f2 = self.rgb_encoder(rgb_f1, 'conv1_2_mp', 'conv2_2_mp')
        rgb_f3 = self.rgb_encoder(rgb_f2, 'conv2_2_mp', 'conv3_3_mp')
        ################

        ### STAGE 2 ###
        rgb_f4 = self.rgb_encoder(rgb_f3, 'conv3_3_mp', 'conv4_3_mp')
        c_rgb_f4 = self.rgb_sq4(rgb_f4)
        
        rgb_f4 = self.DAM4(rgb_f4, c_rgb_f4)

        ### STAGE 3 ###
        rgb_f5 = self.rgb_encoder(rgb_f4, 'conv4_3_mp', 'conv5_3_mp')
        c_rgb_f5 = self.rgb_sq5(rgb_f5)

        da_rgb_f5 = self.DAM5(rgb_f5, c_rgb_f5)
        ###############

        ### STAGE 4 ###
        rgb_cf6 = self.rgb_COMP6(da_rgb_f5)
        rgb_cf6 = self.rgb_aspp(rgb_cf6)
        rgb_cf5 = self.rgb_COMP5(da_rgb_f5)
        rgb_cf4 = self.rgb_COMP4(rgb_f4)
        ###############

        feat_56 = self.rgb_merge_co_56(torch.cat([rgb_cf5, resize(rgb_cf6, [s // 16, s // 16])], dim=1))
        feat_45 = self.rgb_merge_co_45(torch.cat([rgb_cf4, resize(feat_56, [s // 8, s // 8])], dim=1))
        cosal_map_4 = self.rgb_get_pred_4(feat_45)

        feat_34, cosal_map_3 = self.rgb_refine_3(rgb_f3, cosal_map_4, feat_45)
        feat_23, cosal_map_2 = self.rgb_refine_2(rgb_f2, cosal_map_4, feat_34)
        _, cosal_map_1 = self.rgb_refine_1(rgb_f1, cosal_map_4, feat_23)

        preds_list = [resize(cosal_map_4, [s, s]), resize(cosal_map_3, [s, s]), resize(cosal_map_2, [s, s]), cosal_map_1]
        return preds_list


if __name__ == '__main__':
    model = DATA().cuda()
    model.eval()
    rgb = torch.rand(1, 3, 512, 512).cuda()

    total = 0
    for i in range(100):
        start = time.time()
        out = model(rgb, 512)
        end = time.time()

        total += end - start
    
    print(1/ (total / 100))