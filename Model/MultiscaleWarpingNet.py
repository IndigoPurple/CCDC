import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import *
from FlowNet_model import FlowNet
from FlowNet_model import FlowNet_dilation
from Backward_warp_layer import Backward_warp
from SupervisedFlowNet import SupervisedFlowNet
import numpy as np
from PIL import Image



# original crossnet
class Crossnetpp_Original(nn.Module):

    def __init__(self, flownet_type='FlowNet_ori'):
        super(Crossnetpp_Original, self).__init__()

        if flownet_type == 'FlowNet_ori':
            # self.FlowNet_s1 = FlowNet(6)
            self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            # self.FlowNet_s1 = FlowNet_dilation(6)
            self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, buff, require_flow=False, flow_visible=False):
        # input_img1_HR = buff['input_img1_HR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        # input_img2_HR = buff['input_img2_HR'].cuda()
        input_img2_Gray = buff['input_img2_Gray'].cuda()
        # print(input_img1_LR.shape, input_img2_HR.shape)
        # flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        # flow_s1_12_1 = flow_s1['flow_12_1']

        # warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        # warp_img = np.clip(warp_img2_HR.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(warp_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/warp_img.png')

        flow_s2 = self.FlowNet_s2(input_img2_Gray, input_img1_SR)
        # flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        Gray_conv1, Gray_conv2, Gray_conv3, Gray_conv4 = self.Encoder(input_img2_Gray)
        # HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(SR_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(SR_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(SR_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(SR_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_2(Gray_conv1, Gray_conv2, Gray_conv3, Gray_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        # sr_img = np.clip(refSR_2.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(sr_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/sr_img.png')
        # exit()
        if flow_visible:
            return refSR_2, flow_s2
        else:
            if require_flow:
                return refSR_2
            else:
                return refSR_2
        # if flow_visible:
        #     return warp_img2_HR, refSR_2, flow_s1, flow_s2
        # else:
        #     if require_flow:
        #         return warp_img2_HR, refSR_2, flow_s1_12_1
        #     else:
        #         return warp_img2_HR, refSR_2

class ColorNet0(nn.Module):

    def __init__(self, flownet_type='FlowNet_ori'):
        super(ColorNet0, self).__init__()

        if flownet_type == 'FlowNet_ori':
            # self.FlowNet_s1 = FlowNet(6)
            self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            # self.FlowNet_s1 = FlowNet_dilation(6)
            self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_gray = UNet_decoder_gray()

    def forward(self, buff, require_flow=False, flow_visible=False):
        # input_img1_HR = buff['input_img1_HR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        # input_img2_HR = buff['input_img2_HR'].cuda()
        input_img1_Gray = buff['input_img1_Gray'].cuda()
        input_img2_Gray = buff['input_img2_Gray'].cuda()
        # print(input_img1_LR.shape, input_img2_HR.shape)
        # flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        # flow_s1_12_1 = flow_s1['flow_12_1']

        # warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        # warp_img = np.clip(warp_img2_HR.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(warp_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/warp_img.png')

        flow_s2 = self.FlowNet_s2(input_img2_Gray, input_img1_SR)
        # flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        Gray1_conv1, Gray1_conv2, Gray1_conv3, Gray1_conv4 = self.Encoder(input_img1_Gray)
        Gray2_conv1, Gray2_conv2, Gray2_conv3, Gray2_conv4 = self.Encoder(input_img2_Gray)
        # HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)
        warp_gray_conv1 = self.Backward_warp(Gray1_conv1, flow_s2_12_1)
        warp_gray_conv2 = self.Backward_warp(Gray1_conv2, flow_s2_12_2)
        warp_gray_conv3 = self.Backward_warp(Gray1_conv3, flow_s2_12_3)
        warp_gray_conv4 = self.Backward_warp(Gray1_conv4, flow_s2_12_4)
        warp_img1_Gray = self.Backward_warp(input_img1_Gray, flow_s2_12_1)
        warp_s2_21_conv1 = self.Backward_warp(SR_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(SR_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(SR_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(SR_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_gray(Gray2_conv1, Gray2_conv2, Gray2_conv3, Gray2_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4, warp_gray_conv1 - Gray2_conv1,
                                         warp_gray_conv2 - Gray2_conv2, warp_gray_conv3 - Gray2_conv3,
                                         warp_gray_conv4 - Gray2_conv4, warp_img1_Gray - input_img2_Gray)

        # sr_img = np.clip(refSR_2.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(sr_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/sr_img.png')
        # exit()
        if flow_visible:
            return refSR_2, warp_img1_Gray, flow_s2
        else:
            if require_flow:
                return refSR_2, warp_img1_Gray
            else:
                return refSR_2, warp_img1_Gray
        # if flow_visible:
        #     return warp_img2_HR, refSR_2, flow_s1, flow_s2
        # else:
        #     if require_flow:
        #         return warp_img2_HR, refSR_2, flow_s1_12_1
        #     else:
        #         return warp_img2_HR, refSR_2

class ColorNet1(nn.Module):

    def __init__(self, flownet_type='FlowNet_ori'):
        super(ColorNet1, self).__init__()

        if flownet_type == 'FlowNet_ori':
            # self.FlowNet_s1 = FlowNet(6)
            self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            # self.FlowNet_s1 = FlowNet_dilation(6)
            self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_gray = UNet_decoder_gray()

    def forward(self, buff, require_flow=False, flow_visible=False):
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        # input_img2_HR = buff['input_img2_HR'].cuda()
        input_img1_Gray = buff['input_img1_Gray'].cuda()
        input_img2_Gray = buff['input_img2_Gray'].cuda()
        # print(input_img1_LR.shape, input_img2_HR.shape)
        # flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        # flow_s1_12_1 = flow_s1['flow_12_1']

        # warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        # warp_img = np.clip(warp_img2_HR.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(warp_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/warp_img.png')

        flow_s2 = self.FlowNet_s2(input_img2_Gray, input_img1_SR)
        # flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        Gray1_conv1, Gray1_conv2, Gray1_conv3, Gray1_conv4 = self.Encoder(input_img1_Gray)
        Gray2_conv1, Gray2_conv2, Gray2_conv3, Gray2_conv4 = self.Encoder(input_img2_Gray)
        # HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)
        warp_gray_conv1 = self.Backward_warp(Gray1_conv1, flow_s2_12_1)
        warp_gray_conv2 = self.Backward_warp(Gray1_conv2, flow_s2_12_2)
        warp_gray_conv3 = self.Backward_warp(Gray1_conv3, flow_s2_12_3)
        warp_gray_conv4 = self.Backward_warp(Gray1_conv4, flow_s2_12_4)
        warp_img1_Gray = self.Backward_warp(input_img1_Gray, flow_s2_12_1)
        warp_img1_LR = self.Backward_warp(input_img1_LR, flow_s2_12_1)
        warp_s2_21_conv1 = self.Backward_warp(SR_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(SR_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(SR_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(SR_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_gray(Gray2_conv1, Gray2_conv2, Gray2_conv3, Gray2_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4, warp_gray_conv1 - Gray2_conv1,
                                         warp_gray_conv2 - Gray2_conv2, warp_gray_conv3 - Gray2_conv3,
                                         warp_gray_conv4 - Gray2_conv4, warp_img1_Gray - input_img2_Gray)

        # sr_img = np.clip(refSR_2.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(sr_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/sr_img.png')
        # exit()
        # if flow_visible:
        #     return refSR_2, warp_img1_LR, warp_img1_Gray - input_img2_Gray, flow_s2
        # else:
        #     if require_flow:
        #         return refSR_2, warp_img1_LR, warp_img1_Gray - input_img2_Gray
        #     else:
        #         return refSR_2, warp_img1_LR, warp_img1_Gray - input_img2_Gray
        # if flow_visible:
        #     return warp_img2_HR, refSR_2, flow_s1, flow_s2
        # else:
        #     if require_flow:
        #         return warp_img2_HR, refSR_2, flow_s1_12_1
        #     else:
        #         return warp_img2_HR, refSR_2
        if flow_visible:
            return refSR_2
        else:
            if require_flow:
                return refSR_2
            else:
                return refSR_2

class ColorNet2(nn.Module):

    def __init__(self, flownet_type='FlowNet_ori'):
        super(ColorNet2, self).__init__()

        if flownet_type == 'FlowNet_ori':
            # self.FlowNet_s1 = FlowNet(6)
            self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            # self.FlowNet_s1 = FlowNet_dilation(6)
            self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_gray = UNet_decoder_gray()

    def forward(self, buff, require_flow=False, flow_visible=False):
        # input_img1_HR = buff['input_img1_HR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        # input_img2_HR = buff['input_img2_HR'].cuda()
        input_img1_Gray = buff['input_img1_Gray'].cuda()
        input_img2_Gray = buff['input_img2_Gray'].cuda()
        # print(input_img1_LR.shape, input_img2_HR.shape)
        # flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        # flow_s1_12_1 = flow_s1['flow_12_1']

        # warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        # warp_img = np.clip(warp_img2_HR.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(warp_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/warp_img.png')

        flow_s2 = self.FlowNet_s2(input_img2_Gray, input_img1_SR)
        # flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        Gray1_conv1, Gray1_conv2, Gray1_conv3, Gray1_conv4 = self.Encoder(input_img1_Gray)
        Gray2_conv1, Gray2_conv2, Gray2_conv3, Gray2_conv4 = self.Encoder(input_img2_Gray)
        # HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)
        warp_gray_conv1 = self.Backward_warp(Gray1_conv1, flow_s2_12_1)
        warp_gray_conv2 = self.Backward_warp(Gray1_conv2, flow_s2_12_2)
        warp_gray_conv3 = self.Backward_warp(Gray1_conv3, flow_s2_12_3)
        warp_gray_conv4 = self.Backward_warp(Gray1_conv4, flow_s2_12_4)
        warp_img1_Gray = self.Backward_warp(input_img1_Gray, flow_s2_12_1)
        warp_img1_RGB = self.Backward_warp(input_img1_SR, flow_s2_12_1)
        warp_s2_21_conv1 = self.Backward_warp(SR_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(SR_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(SR_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(SR_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_gray(Gray2_conv1, Gray2_conv2, Gray2_conv3, Gray2_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4, warp_gray_conv1 - Gray2_conv1,
                                         warp_gray_conv2 - Gray2_conv2, warp_gray_conv3 - Gray2_conv3,
                                         warp_gray_conv4 - Gray2_conv4, warp_img1_Gray - input_img2_Gray)

        # sr_img = np.clip(refSR_2.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(sr_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/sr_img.png')
        # exit()
        if flow_visible:
            return refSR_2, warp_img1_RGB, flow_s2
        else:
            if require_flow:
                return refSR_2, warp_img1_RGB
            else:
                return refSR_2, warp_img1_RGB
        # if flow_visible:
        #     return warp_img2_HR, refSR_2, flow_s1, flow_s2
        # else:
        #     if require_flow:
        #         return warp_img2_HR, refSR_2, flow_s1_12_1
        #     else:
        #         return warp_img2_HR, refSR_2

class CocoNet0(nn.Module):
# 1 stage
    def __init__(self, flownet_type='FlowNet_ori'):
        super(CocoNet0, self).__init__()

        if flownet_type == 'FlowNet_ori':
            self.FlowNet_s1 = FlowNet(6)
            # self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            self.FlowNet_s1 = FlowNet_dilation(6)
            # self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_s1 = UNet_decoder_2()
        # self.UNet_decoder_s2 = UNet_decoder_2()

    def forward(self, buff, require_flow=False, flow_visible=False):
        # input_img1_HR = buff['input_img1_HR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        # input_img2_HR = buff['input_img2_HR'].cuda()
        input_img2_Gray = buff['input_img2_Gray'].cuda()
        # print(input_img1_LR.shape, input_img2_HR.shape)
        # flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        # flow_s1_12_1 = flow_s1['flow_12_1']

        # warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        # warp_img = np.clip(warp_img2_HR.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(warp_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/warp_img.png')
        '''1111111111111111111111111111color'''
        flow = self.FlowNet_s1(input_img2_Gray, input_img1_SR) # flow from GrayHR to RGBLR


        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        Gray_conv1, Gray_conv2, Gray_conv3, Gray_conv4 = self.Encoder(input_img2_Gray)
        # HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s1_21_conv1 = self.Backward_warp(SR_conv1, flow['flow_12_1'])
        warp_s1_21_conv2 = self.Backward_warp(SR_conv2, flow['flow_12_2'])
        warp_s1_21_conv3 = self.Backward_warp(SR_conv3, flow['flow_12_3'])
        warp_s1_21_conv4 = self.Backward_warp(SR_conv4, flow['flow_12_4'])
        img2_color1 = self.UNet_decoder_s1(Gray_conv1, Gray_conv2, Gray_conv3, Gray_conv4, warp_s1_21_conv1, warp_s1_21_conv2,
                                      warp_s1_21_conv3, warp_s1_21_conv4)
        '''222222222222222222222222222222SR'''
        flow = self.FlowNet_s1(input_img1_SR, img2_color1) # flow from RGBLR to GrayHR

        # gray_conv1, gray_conv2, gray_conv3, gray_conv4 = self.Encoder(input_img2_Gray)
        warp_s2_21_conv1 = self.Backward_warp(Gray_conv1, flow['flow_12_1'])
        warp_s2_21_conv2 = self.Backward_warp(Gray_conv2, flow['flow_12_2'])
        warp_s2_21_conv3 = self.Backward_warp(Gray_conv3, flow['flow_12_3'])
        warp_s2_21_conv4 = self.Backward_warp(Gray_conv4, flow['flow_12_4'])
        img1_refSR1 = self.UNet_decoder_s1(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1,
                                          warp_s2_21_conv2,
                                          warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return img1_refSR1, img2_color1, flow
        else:
            if require_flow:
                return img1_refSR1, img2_color1
            else:
                return img1_refSR1, img2_color1

class CocoNet1(nn.Module):
# 2 stage
    def __init__(self, flownet_type='FlowNet_ori'):
        super(CocoNet1, self).__init__()
        print('66666666666666666666666666666666666666666666666666666666666')
        if flownet_type == 'FlowNet_ori':
            self.FlowNet_s1 = FlowNet(6)
            # self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            self.FlowNet_s1 = FlowNet_dilation(6)
            # self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_s1 = UNet_decoder_2()
        self.UNet_decoder_s2 = UNet_decoder_res()

    def forward(self, buff, require_flow=False, flow_visible=False):
        # input_img1_HR = buff['input_img1_HR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        # input_img2_HR = buff['input_img2_HR'].cuda()
        input_img2_Gray = buff['input_img2_Gray'].cuda()
        # print(input_img1_LR.shape, input_img2_HR.shape)
        # flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        # flow_s1_12_1 = flow_s1['flow_12_1']

        # warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        # warp_img = np.clip(warp_img2_HR.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(warp_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/warp_img.png')
        '''1111111111111111111111111111color'''
        flow = self.FlowNet_s1(input_img2_Gray, input_img1_SR) # flow from GrayHR to RGBLR


        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        Gray_conv1, Gray_conv2, Gray_conv3, Gray_conv4 = self.Encoder(input_img2_Gray)
        # HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s1_21_conv1 = self.Backward_warp(SR_conv1, flow['flow_12_1'])
        warp_s1_21_conv2 = self.Backward_warp(SR_conv2, flow['flow_12_2'])
        warp_s1_21_conv3 = self.Backward_warp(SR_conv3, flow['flow_12_3'])
        warp_s1_21_conv4 = self.Backward_warp(SR_conv4, flow['flow_12_4'])
        img2_color1 = self.UNet_decoder_s1(Gray_conv1, Gray_conv2, Gray_conv3, Gray_conv4, warp_s1_21_conv1, warp_s1_21_conv2,
                                      warp_s1_21_conv3, warp_s1_21_conv4)
        '''222222222222222222222222222222SR'''
        flow = self.FlowNet_s1(input_img1_SR, img2_color1) # flow from RGBLR to GrayHR

        color_conv1, color_conv2, color_conv3, color_conv4 = self.Encoder(img2_color1)
        warp_s2_21_conv1 = self.Backward_warp(color_conv1, flow['flow_12_1'])
        warp_s2_21_conv2 = self.Backward_warp(color_conv2, flow['flow_12_2'])
        warp_s2_21_conv3 = self.Backward_warp(color_conv3, flow['flow_12_3'])
        warp_s2_21_conv4 = self.Backward_warp(color_conv4, flow['flow_12_4'])
        img1_refSR1 = self.UNet_decoder_s1(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1,
                                          warp_s2_21_conv2,
                                          warp_s2_21_conv3, warp_s2_21_conv4)
        '''333333333333333333333333333333color'''
        flow = self.FlowNet_s1(img2_color1, img1_refSR1) # flow from GrayHR to RGBLR

        SR1_conv1, SR1_conv2, SR1_conv3, SR1_conv4 = self.Encoder(img1_refSR1)

        warp_s1_21_conv1 = self.Backward_warp(SR1_conv1, flow['flow_12_1'])
        warp_s1_21_conv2 = self.Backward_warp(SR1_conv2, flow['flow_12_2'])
        warp_s1_21_conv3 = self.Backward_warp(SR1_conv3, flow['flow_12_3'])
        warp_s1_21_conv4 = self.Backward_warp(SR1_conv4, flow['flow_12_4'])
        res1 = self.UNet_decoder_s2(warp_s1_21_conv1 - color_conv1, warp_s1_21_conv2 - color_conv2, warp_s1_21_conv3 - color_conv3, warp_s1_21_conv4 - color_conv4)
        img2_color2 = img2_color1 + res1
        '''444444444444444444444444444SR'''
        flow = self.FlowNet_s1(input_img1_SR, img2_color1)  # flow from RGBLR to GrayHR
        color2_conv1, color2_conv2, color2_conv3, color2_conv4 = self.Encoder(img2_color2)
        warp_s2_21_conv1 = self.Backward_warp(color2_conv1, flow['flow_12_1'])
        warp_s2_21_conv2 = self.Backward_warp(color2_conv2, flow['flow_12_2'])
        warp_s2_21_conv3 = self.Backward_warp(color2_conv3, flow['flow_12_3'])
        warp_s2_21_conv4 = self.Backward_warp(color2_conv4, flow['flow_12_4'])
        res2 = self.UNet_decoder_s2(warp_s2_21_conv1 - SR_conv1, warp_s2_21_conv2 - SR_conv2, warp_s2_21_conv3 - SR_conv3, warp_s2_21_conv4 - SR_conv4)
        img1_refSR2 = img1_refSR1 + res2

        if flow_visible:
            return img1_refSR2, res1, res2, img2_color2, flow
        else:
            if require_flow:
                return img1_refSR2, res1, res2, img2_color2
            else:
                return img1_refSR2, res1, res2, img2_color2

class CocoNet2(nn.Module):
# 2 stage
    def __init__(self, flownet_type='FlowNet_ori'):
        super(CocoNet2, self).__init__()

        if flownet_type == 'FlowNet_ori':
            self.FlowNet_s1 = FlowNet(6)
            # self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            self.FlowNet_s1 = FlowNet_dilation(6)
            # self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_s1 = UNet_decoder_2()
        # self.UNet_decoder_s2 = UNet_decoder_res()

    def forward(self, buff, require_flow=False, flow_visible=False):
        # input_img1_HR = buff['input_img1_HR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        # input_img2_HR = buff['input_img2_HR'].cuda()
        input_img2_Gray = buff['input_img2_Gray'].cuda()
        # print(input_img1_LR.shape, input_img2_HR.shape)
        # flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        # flow_s1_12_1 = flow_s1['flow_12_1']

        # warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        # warp_img = np.clip(warp_img2_HR.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(warp_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/warp_img.png')
        '''1111111111111111111111111111color'''
        flow = self.FlowNet_s1(input_img2_Gray, input_img1_SR) # flow from GrayHR to RGBLR


        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        Gray_conv1, Gray_conv2, Gray_conv3, Gray_conv4 = self.Encoder(input_img2_Gray)
        # HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s1_21_conv1 = self.Backward_warp(SR_conv1, flow['flow_12_1'])
        warp_s1_21_conv2 = self.Backward_warp(SR_conv2, flow['flow_12_2'])
        warp_s1_21_conv3 = self.Backward_warp(SR_conv3, flow['flow_12_3'])
        warp_s1_21_conv4 = self.Backward_warp(SR_conv4, flow['flow_12_4'])
        img2_color1 = self.UNet_decoder_s1(Gray_conv1, Gray_conv2, Gray_conv3, Gray_conv4, warp_s1_21_conv1, warp_s1_21_conv2,
                                      warp_s1_21_conv3, warp_s1_21_conv4)
        '''222222222222222222222222222222SR'''
        flow = self.FlowNet_s1(input_img1_SR, img2_color1) # flow from RGBLR to GrayHR

        color_conv1, color_conv2, color_conv3, color_conv4 = self.Encoder(img2_color1)
        warp_s2_21_conv1 = self.Backward_warp(color_conv1, flow['flow_12_1'])
        warp_s2_21_conv2 = self.Backward_warp(color_conv2, flow['flow_12_2'])
        warp_s2_21_conv3 = self.Backward_warp(color_conv3, flow['flow_12_3'])
        warp_s2_21_conv4 = self.Backward_warp(color_conv4, flow['flow_12_4'])
        img1_refSR1 = self.UNet_decoder_s1(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1,
                                          warp_s2_21_conv2,
                                          warp_s2_21_conv3, warp_s2_21_conv4)
        '''333333333333333333333333333333color'''
        flow = self.FlowNet_s1(img2_color1, img1_refSR1) # flow from GrayHR to RGBLR

        SR1_conv1, SR1_conv2, SR1_conv3, SR1_conv4 = self.Encoder(img1_refSR1)

        warp_s1_21_conv1 = self.Backward_warp(SR1_conv1, flow['flow_12_1'])
        warp_s1_21_conv2 = self.Backward_warp(SR1_conv2, flow['flow_12_2'])
        warp_s1_21_conv3 = self.Backward_warp(SR1_conv3, flow['flow_12_3'])
        warp_s1_21_conv4 = self.Backward_warp(SR1_conv4, flow['flow_12_4'])
        res = self.UNet_decoder_s1(color_conv1, color_conv2, color_conv3, color_conv4,  warp_s1_21_conv1, warp_s1_21_conv2,
                                      warp_s1_21_conv3, warp_s1_21_conv4)
        img2_color2 = img2_color1 + res
        '''444444444444444444444444444SR'''
        flow = self.FlowNet_s1(input_img1_SR, img2_color1)  # flow from RGBLR to GrayHR
        color2_conv1, color2_conv2, color2_conv3, color2_conv4 = self.Encoder(img2_color2)
        warp_s2_21_conv1 = self.Backward_warp(color2_conv1, flow['flow_12_1'])
        warp_s2_21_conv2 = self.Backward_warp(color2_conv2, flow['flow_12_2'])
        warp_s2_21_conv3 = self.Backward_warp(color2_conv3, flow['flow_12_3'])
        warp_s2_21_conv4 = self.Backward_warp(color2_conv4, flow['flow_12_4'])
        res = self.UNet_decoder_s1(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1,
                                          warp_s2_21_conv2,
                                          warp_s2_21_conv3, warp_s2_21_conv4)
        img1_refSR2 = img1_refSR1 + res

        if flow_visible:
            return img1_refSR2, img2_color2, flow
        else:
            if require_flow:
                return img1_refSR2, img2_color2
            else:
                return img1_refSR2, img2_color2

class CocoNet3(nn.Module):
# 2 stage
    def __init__(self, flownet_type='FlowNet_ori'):
        super(CocoNet3, self).__init__()

        if flownet_type == 'FlowNet_ori':
            self.FlowNet_s1 = FlowNet(6)
            self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            self.FlowNet_s1 = FlowNet_dilation(6)
            self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_s1 = UNet_decoder_2()
        self.UNet_decoder_s2 = UNet_decoder_2()
        self.UNet_decoder_res = UNet_decoder_res()

    def forward(self, buff, require_flow=False, flow_visible=False):
        # input_img1_HR = buff['input_img1_HR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        # input_img2_HR = buff['input_img2_HR'].cuda()
        input_img2_Gray = buff['input_img2_Gray'].cuda()
        # print(input_img1_LR.shape, input_img2_HR.shape)
        # flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        # flow_s1_12_1 = flow_s1['flow_12_1']

        # warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        # warp_img = np.clip(warp_img2_HR.cpu().numpy(), 0.0, 1.0)
        # img = Image.fromarray(np.array(warp_img[0].transpose(1, 2, 0) * 255, dtype=np.uint8))
        # img.save('./result/debug/vimeo_original_0909/warp_img.png')
        '''1111111111111111111111111111color'''
        flow = self.FlowNet_s1(input_img2_Gray, input_img1_SR) # flow from GrayHR to RGBLR


        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        Gray_conv1, Gray_conv2, Gray_conv3, Gray_conv4 = self.Encoder(input_img2_Gray)
        # HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s1_21_conv1 = self.Backward_warp(SR_conv1, flow['flow_12_1'])
        warp_s1_21_conv2 = self.Backward_warp(SR_conv2, flow['flow_12_2'])
        warp_s1_21_conv3 = self.Backward_warp(SR_conv3, flow['flow_12_3'])
        warp_s1_21_conv4 = self.Backward_warp(SR_conv4, flow['flow_12_4'])
        img2_color1 = self.UNet_decoder_s1(Gray_conv1, Gray_conv2, Gray_conv3, Gray_conv4, warp_s1_21_conv1, warp_s1_21_conv2,
                                      warp_s1_21_conv3, warp_s1_21_conv4)
        '''222222222222222222222222222222SR'''
        flow = self.FlowNet_s2(input_img1_SR, img2_color1) # flow from RGBLR to GrayHR

        color_conv1, color_conv2, color_conv3, color_conv4 = self.Encoder(img2_color1)
        warp_s2_21_conv1 = self.Backward_warp(color_conv1, flow['flow_12_1'])
        warp_s2_21_conv2 = self.Backward_warp(color_conv2, flow['flow_12_2'])
        warp_s2_21_conv3 = self.Backward_warp(color_conv3, flow['flow_12_3'])
        warp_s2_21_conv4 = self.Backward_warp(color_conv4, flow['flow_12_4'])
        img1_refSR1 = self.UNet_decoder_s2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1,
                                          warp_s2_21_conv2,
                                          warp_s2_21_conv3, warp_s2_21_conv4)
        '''333333333333333333333333333333color'''
        flow = self.FlowNet_s2(img2_color1, img1_refSR1) # flow from GrayHR to RGBLR

        SR1_conv1, SR1_conv2, SR1_conv3, SR1_conv4 = self.Encoder(img1_refSR1)

        warp_s1_21_conv1 = self.Backward_warp(SR1_conv1, flow['flow_12_1'])
        warp_s1_21_conv2 = self.Backward_warp(SR1_conv2, flow['flow_12_2'])
        warp_s1_21_conv3 = self.Backward_warp(SR1_conv3, flow['flow_12_3'])
        warp_s1_21_conv4 = self.Backward_warp(SR1_conv4, flow['flow_12_4'])
        res = self.UNet_decoder_res(warp_s1_21_conv1 - color_conv1, warp_s1_21_conv2 - color_conv2, warp_s1_21_conv3 - color_conv3, warp_s1_21_conv4 - color_conv4)
        img2_color2 = img2_color1 + res
        '''444444444444444444444444444SR'''
        flow = self.FlowNet_s2(input_img1_SR, img2_color1)  # flow from RGBLR to GrayHR
        color2_conv1, color2_conv2, color2_conv3, color2_conv4 = self.Encoder(img2_color2)
        warp_s2_21_conv1 = self.Backward_warp(color2_conv1, flow['flow_12_1'])
        warp_s2_21_conv2 = self.Backward_warp(color2_conv2, flow['flow_12_2'])
        warp_s2_21_conv3 = self.Backward_warp(color2_conv3, flow['flow_12_3'])
        warp_s2_21_conv4 = self.Backward_warp(color2_conv4, flow['flow_12_4'])
        res = self.UNet_decoder_res(warp_s2_21_conv1 - SR_conv1, warp_s2_21_conv2 - SR_conv2, warp_s2_21_conv3 - SR_conv3, warp_s2_21_conv4 - SR_conv4)
        img1_refSR2 = img1_refSR1 + res

        if flow_visible:
            return img1_refSR2, img2_color2, flow
        else:
            if require_flow:
                return img1_refSR2, img2_color2
            else:
                return img1_refSR2, img2_color2

class Crossnetpp_Multiflow1(nn.Module):

    def __init__(self):
        super(Crossnetpp_Multiflow1, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=7):

        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        input_img2_HR = buff['input_img2_HR'].cuda()

        warp_img2_HR_list = tuple()
        for i in range(frame_num):
            flow = self.FlowNet_s1(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_12_1 = flow['flow_12_1']
            input_img2_HR = self.Backward_warp(input_img2_HR, flow_12_1)
            warp_img2_HR_list = warp_img2_HR_list + (input_img2_HR,)

        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)
        warp_img2_HR = input_img2_HR

        flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

class Crossnetpp_Multiflow2(nn.Module):
    def __init__(self):
        super(Crossnetpp_Multiflow2, self).__init__()
        self.FlowNet_LR = FlowNet(6)
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        # self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        # self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=7):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_LR_list = tuple()
        warp_img2_HR_list = tuple()
        flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num - 1) * 3: frame_num * 3, :, :], input_img2_HR)
        flow_s1_12_1 = flow_s1['flow_12_1']
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR,)
        flow_LR_list = flow_LR_list + (flow_s1_12_1,)
        for i in range(frame_num - 1):
            flow_LR = self.FlowNet_LR(input_LR[:, (frame_num - (i + 2)) * 3: (frame_num - (i + 1)) * 3, :, :],
                                      input_LR[:, (frame_num - (i + 1)) * 3: (frame_num - i) * 3, :, :])
            flow_LR_1 = flow_LR['flow_12_1']
            warp_img2_HR = self.Backward_warp(warp_img2_HR, flow_LR_1)
            flow_LR_list = flow_LR_list + (flow_LR_1,)
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        # ########## flow fusion
        # flow_LR_list_tensor = torch.cat(flow_LR_list, dim=1)
        # # print(flow_12_list[0].size(), flow_12_list.size())
        # flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_LR_list_tensor)
        # flow_s1_final = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)

        ########## warp
        # warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)
        flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1_12_1
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

class Crossnetpp_MultiflowFusion1(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion1, self).__init__()
        self.FlowNet_LR = FlowNet(6)
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=7):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_LR_list = tuple()
        warp_img2_HR_list = tuple()
        flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num - 1) * 3: frame_num * 3, :, :], input_img2_HR)
        flow_s1_12_1 = flow_s1['flow_12_1']
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR,)
        flow_LR_list = flow_LR_list + (flow_s1_12_1,)
        for i in range(frame_num - 1):
            flow_LR = self.FlowNet_LR(input_LR[:, (frame_num - (i + 2)) * 3: (frame_num - (i + 1)) * 3, :, :],
                                      input_LR[:, (frame_num - (i + 1)) * 3: (frame_num - i) * 3, :, :])
            flow_LR_1 = flow_LR['flow_12_1']
            warp_img2_HR = self.Backward_warp(warp_img2_HR, flow_LR_1)
            flow_LR_list = flow_LR_list + (flow_LR_1,)
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_LR_list_tensor = torch.cat(flow_LR_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_LR_list_tensor)
        flow_residue = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)
        flow_s1_final = flow_residue + flow_s1_12_1

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)
        flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1_12_1
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

class Crossnetpp_MultiflowFusion2(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion2, self).__init__()
        self.FlowNet_LR = FlowNet(6)
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=7):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_LR_list = tuple()
        warp_img2_HR_list = tuple()
        flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num - 1) * 3: frame_num * 3, :, :], input_img2_HR)
        flow_s1_12_1 = flow_s1['flow_12_1']
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)
        warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR,)
        flow_LR_list = flow_LR_list + (flow_s1_12_1,)
        for i in range(frame_num - 1):
            flow_LR = self.FlowNet_LR(input_LR[:, (frame_num - (i + 2)) * 3: (frame_num - (i + 1)) * 3, :, :],
                                      input_LR[:, (frame_num - (i + 1)) * 3: (frame_num - i) * 3, :, :])
            flow_LR_1 = flow_LR['flow_12_1']
            warp_img2_HR = self.Backward_warp(warp_img2_HR, flow_LR_1)
            flow_LR_list = flow_LR_list + (flow_LR_1,)
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_LR_list_tensor = torch.cat(flow_LR_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_LR_list_tensor)
        flow_s1_final = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)
        flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1_12_1
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

class Crossnetpp_MultiflowFusion3(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion3, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=7):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_s1_12_list = tuple()
        warp_img2_HR_list = tuple()
        for i in range(frame_num):
            flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_s1_12_list = flow_s1_12_list + (flow_s1['flow_12_1'],)
            warp_img2_HR_temp = self.Backward_warp(input_img2_HR, flow_s1['flow_12_1'])
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_s1_12_list_tensor = torch.cat(flow_s1_12_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_s1_12_list_tensor)
        flow_residue = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)
        flow_s1_final = flow_residue + flow_s1_12_list[-1]

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)

        flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1['flow_12_1']
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

class Crossnetpp_MultiflowFusion4(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion4, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=7):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_s1_12_list = tuple()
        warp_img2_HR_list = tuple()
        for i in range(frame_num):
            flow_s1 = self.FlowNet_s1(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_s1_12_list = flow_s1_12_list + (flow_s1['flow_12_1'],)
            warp_img2_HR_temp = self.Backward_warp(input_img2_HR, flow_s1['flow_12_1'])
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_s1_12_list_tensor = torch.cat(flow_s1_12_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_s1_12_list_tensor)
        flow_s1_final = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)

        flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1['flow_12_1']
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

class Crossnetpp_MultiflowFusion4_1(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion4_1, self).__init__()
        # self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(14)   # 14 = 2*7
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=7):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        input_img2_HR = buff['input_img2_HR'].cuda()

        flow_s1_12_list = tuple()
        warp_img2_HR_list = tuple()
        for i in range(frame_num):
            flow_s1 = self.FlowNet_s2(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_s1_12_list = flow_s1_12_list + (flow_s1['flow_12_1'],)
            warp_img2_HR_temp = self.Backward_warp(input_img2_HR, flow_s1['flow_12_1'])
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_s1_12_list_tensor = torch.cat(flow_s1_12_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_s1_12_list_tensor)
        flow_s1_final = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)

        flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1['flow_12_1']
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

class Crossnetpp_MultiflowFusion5(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion5, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(26)   # 26 = 2*(7+6)
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=7):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        input_img2_HR = buff['input_img2_HR'].cuda()

        # flow_LR_list = tuple()
        flow_s1_12_list = tuple()
        warp_img2_HR_list = tuple()
        flow_s1_LR = self.FlowNet_s2(input_LR[:, (frame_num - 1) * 3: frame_num * 3, :, :], input_img2_HR)
        flow_s1_12_1_LR = flow_s1_LR['flow_12_1']
        warp_img2_HR_temp = self.Backward_warp(input_img2_HR, flow_s1_12_1_LR)
        warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        # flow_s1_12_list = flow_s1_12_list + (flow_s1_12_1_LR,)
        for i in range(frame_num-1):
            '''LR--LR'''
            flow_LR = self.FlowNet_s1(input_LR[:, (frame_num - (i + 2)) * 3: (frame_num - (i + 1)) * 3, :, :],
                                      input_LR[:, (frame_num - (i + 1)) * 3: (frame_num - i) * 3, :, :])
            flow_LR_1 = flow_LR['flow_12_1']
            warp_img2_HR_temp = self.Backward_warp(warp_img2_HR_temp, flow_LR_1)
            flow_s1_12_list = flow_s1_12_list + (flow_LR_1,)
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        for i in range(frame_num):
            '''REF--LR'''
            flow_s1 = self.FlowNet_s2(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_s1_12_list = flow_s1_12_list + (flow_s1['flow_12_1'],)
            # warp_img2_HR_temp = self.Backward_warp(input_img2_HR, flow_s1['flow_12_1'])
            # warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_s1_12_list_tensor = torch.cat(flow_s1_12_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_s1_12_list_tensor)
        flow_residue = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)
        flow_s1_final = flow_residue + flow_s1_12_list[-1]

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)

        flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1['flow_12_1']
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor

class Crossnetpp_MultiflowFusion6(nn.Module):
    def __init__(self):
        super(Crossnetpp_MultiflowFusion6, self).__init__()
        self.FlowNet_s1 = FlowNet(6)
        self.FlowNet_s2 = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        self.Flow_Encoder = Encoder(26)   # 26 = 2*(7+6)
        self.Flow_UNet_decoder = UNet_decoder_3(out_channel=2)

    def forward(self, buff, flow_visible=False, require_flow=False, frame_num=7):
        input_LR = buff['input_LR'].cuda()
        input_img1_LR = buff['input_img1_LR'].cuda()
        input_img1_SR = buff['input_img1_SR'].cuda()
        input_img2_HR = buff['input_img2_HR'].cuda()

        # flow_LR_list = tuple()
        flow_s1_12_list = tuple()
        warp_img2_HR_list = tuple()
        flow_s1_LR = self.FlowNet_s2(input_LR[:, (frame_num - 1) * 3: frame_num * 3, :, :], input_img2_HR)
        flow_s1_12_1_LR = flow_s1_LR['flow_12_1']
        warp_img2_HR_temp = self.Backward_warp(input_img2_HR, flow_s1_12_1_LR)
        warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        # flow_s1_12_list = flow_s1_12_list + (flow_s1_12_1_LR,)
        for i in range(frame_num-1):
            '''LR--LR'''
            flow_LR = self.FlowNet_s1(input_LR[:, (frame_num - (i + 2)) * 3: (frame_num - (i + 1)) * 3, :, :],
                                      input_LR[:, (frame_num - (i + 1)) * 3: (frame_num - i) * 3, :, :])
            flow_LR_1 = flow_LR['flow_12_1']
            warp_img2_HR_temp = self.Backward_warp(warp_img2_HR_temp, flow_LR_1)
            flow_s1_12_list = flow_s1_12_list + (flow_LR_1,)
            warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        for i in range(frame_num):
            '''REF--LR'''
            flow_s1 = self.FlowNet_s2(input_LR[:, (frame_num-(i+1)) * 3: (frame_num-i) * 3, :, :], input_img2_HR)
            flow_s1_12_list = flow_s1_12_list + (flow_s1['flow_12_1'],)
            # warp_img2_HR_temp = self.Backward_warp(input_img2_HR, flow_s1['flow_12_1'])
            # warp_img2_HR_list = warp_img2_HR_list + (warp_img2_HR_temp,)
        warp_img2_HR_list_tensor = torch.cat(warp_img2_HR_list, dim=1)

        ########## flow fusion
        flow_s1_12_list_tensor = torch.cat(flow_s1_12_list, dim=1)
        # print(flow_12_list[0].size(), flow_12_list.size())
        flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4 = self.Flow_Encoder(flow_s1_12_list_tensor)
        flow_s1_final = self.Flow_UNet_decoder(flowfusion_conv1, flowfusion_conv2, flowfusion_conv3, flowfusion_conv4)

        ########## warp
        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_final)

        flow_s2 = self.FlowNet_s2(input_img1_SR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        ######### CrossNet
        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(input_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)

        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1, warp_s2_21_conv2,
                                      warp_s2_21_conv3, warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR, refSR_2, flow_s1, flow_s2
        else:
            if require_flow:
                return warp_img2_HR, refSR_2, flow_s1['flow_12_1']
            else:
                return warp_img2_HR, refSR_2, warp_img2_HR_list_tensor