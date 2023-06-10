import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class FloatingRegionScore(nn.Module):

    def __init__(self, mIoU, in_channels=19, padding_mode='zeros', size=33):
        """
        purity_conv: size*size
        entropy_conv: size*size
        """
        super(FloatingRegionScore, self).__init__()
        self.in_channels = in_channels
        assert size % 2 == 1, "error size"
        self.purity_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=size,
                                     stride=1, padding=int(size / 2), bias=False,
                                     padding_mode=padding_mode, groups=in_channels)
        weight = torch.ones((size, size), dtype=torch.float32)
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        weight = weight.repeat([in_channels, 1, 1, 1])
        weight = nn.Parameter(weight)
        self.purity_conv.weight = weight
        self.purity_conv.requires_grad_(False)

        self.entropy_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size,
                                      stride=1, padding=int(size / 2), bias=False,
                                      padding_mode=padding_mode)
        weight = torch.ones((size, size), dtype=torch.float32)
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        weight = nn.Parameter(weight)
        self.entropy_conv.weight = weight
        self.entropy_conv.requires_grad_(False)

        # At the end of the evaluation, a dictionary of mIoU calculated weights is obtained
        mIoUs = mIoU
        ratios = []
        for index, mIoU in enumerate(mIoUs):
            ratio = 1 / ((mIoUs[index] / (np.sum(mIoUs) + 1e-6)) + 1e-6)
            ratios.append(ratio)
        r_max = np.max(ratios)
        r_min = np.min(ratios)
        self.weights = {}
        for i, r in enumerate(ratios):
            self.weights[i] = (r - r_min + 1e-6) / (r_max - r_min + 1e-6)

    def forward(self, logit):
        """
        return:
            score, purity, entropy
        """
        logit = logit.squeeze(dim=0)  # [19, h ,w]
        p = torch.softmax(logit, dim=0)  # [19, h, w]
        predict = torch.argmax(p, dim=0)  # [h, w]

        # prediction uncertainty
        one_hot = F.one_hot(predict, num_classes=self.in_channels).float()
        one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)  # [1, 19, h, w]
        summary = self.purity_conv(one_hot)  # [1, 19, h, w]
        count = torch.sum(summary, dim=1, keepdim=True)  # [1, 1, h, w]
        dist = summary / count  # [1, 19, h, w]
        region_impurity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True) / math.log(19)  # [1, 1, h, w]

        # Different weights are given based on classification results
        # numpy
        pixel_entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)  # [h w]
        wt_pixel_entropy = torch.zeros_like(pixel_entropy)
        # tensor to numpy
        pixel_entropy_arr = pixel_entropy.cpu().numpy()
        wt_pixel_entropy_arr = wt_pixel_entropy.cpu().numpy()
        predict_arr = predict.cpu().numpy()
        start_wt_time = time.time()
        for i in range(predict_arr.shape[0]):
            for j in range(predict_arr.shape[1]):
                wt_pixel_entropy_arr[i][j] = pixel_entropy_arr[i][j] * self.weights[predict_arr[i][j]]
        end_wt_time = time.time()
        print("wt_time:{}".format(end_wt_time - start_wt_time))

        # numpy to tensor
        wt_pixel_entropy = torch.from_numpy(wt_pixel_entropy_arr).to("cuda", non_blocking=True)

        wt_pixel_entropy = wt_pixel_entropy.unsqueeze(dim=0).unsqueeze(dim=0) / math.log(19)
        region_sum_entropy = self.entropy_conv(wt_pixel_entropy)  # [1, 1, h, w]  经过卷积得到最后的结果
        prediction_uncertainty = region_sum_entropy / count  # [1, 1, h, w]

        score = region_impurity * prediction_uncertainty

        return score.squeeze(dim=0).squeeze(dim=0), region_impurity.squeeze(dim=0).squeeze(
            dim=0), prediction_uncertainty.squeeze(dim=0).squeeze(dim=0)
