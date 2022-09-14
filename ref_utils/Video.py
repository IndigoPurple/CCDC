import numpy as np
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import cv2

class VideoDataset(Dataset):
    ''' Vimeo Dataset '''

    def __init__(self, video_path, transform=None, is_train=True, require_seqid=False):
        # assert frame_window_size%2==1, "frame_window_size should be odd"
        ## file list
        self.cap = cv2.VideoCapture(video_path)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w0, self.h0 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # for i in range(self.count // 2):
        #     _, frame = self.cap.read()
        _, frame = self.cap.read()
        self.w, self.h = (self.w0 // 32) * 32, (self.h0 // 32) * 32
        self.first_frame = Image.fromarray(frame[:self.h, :self.w, :])
        resize = cv2.resize(frame, (int(self.w / 4), int(self.h / 4)), interpolation=cv2.INTER_CUBIC)
        resize = cv2.resize(resize, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        self.resize = Image.fromarray(resize)
        # self.resize = self.first_frame
        self.transform = transform
        self.crop = None
        self.is_train = is_train
        self.require_seqid = require_seqid
        self.video_path = video_path

    def get_list(self, data_path):
        folder_list = list()
        with open(self.data_list_file, 'r') as f_index:
            reader = csv.reader(f_index)
            for row in reader:
                if row:
                    folder_list.append(data_path + row[0] + '/')
        return folder_list

    def get_frame(self, m, f, mode='LR'):
        ''' return a 3-D [3,H,W] float32 [0.0-1.0] tensor '''
        # print (len(self.file_list),f)
        if mode == 'HR':
            filename = self.folder_list_clean[m] + self.file_list[f]
        elif mode == 'LR':
            filename = self.folder_list_corrupted[m] + self.file_list[f]
        elif mode == 'SR':
            filename = self.folder_list_MDSR[m] + self.file_list[f]
        image = Image.open(filename)
        return image

    def get_all_frames(self, m, mode='LR'):
        ''' return a 3-D [3,H,W] float32 [0.0-1.0] tensor '''
        # print (len(self.file_list),f)
        # print('get all frames>>>>>>>>>>')
        if mode == 'HR':
            images_list = tuple()
            for f in self.file_list:
                filename = self.folder_list_clean[m] + f
                image = cv2.imread(filename)
                images_list = images_list + (image,)
            images = np.concatenate(images_list, axis=2)
        elif mode == 'LR':
            images_list = tuple()
            for f in self.file_list:
                filename = self.folder_list_corrupted[m] + f
                image = cv2.imread(filename)
                images_list = images_list + (image,)
            images = np.concatenate(images_list, axis=2)
        elif mode == 'SR':
            images_list = tuple()
            for f in self.file_list:
                filename = self.folder_list_MDSR[m] + f
                image = cv2.imread(filename)
                images = np.concatenate(images_list, axis=2)
            images = np.concatenate(images_list, axis=2)
        return images

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        for i in range(idx+1):
            _, frame = cap.read()
        frame = Image.fromarray(frame[:self.h, :self.w, :])
        sample = dict()

        sample['input_img1_LR'] = self.resize
        if self.transform:
            sample['input_img1_LR'] = self.transform(sample['input_img1_LR'])

        sample['input_img1_Gray'] = self.first_frame.convert('L').convert('RGB')
        if self.transform:
            sample['input_img1_Gray'] = self.transform(sample['input_img1_Gray'])

        sample['input_img1_SR'] = self.resize
        if self.transform:
            sample['input_img1_SR'] = self.transform(sample['input_img1_SR'])

        sample['input_img2_Gray'] = frame.convert('L').convert('RGB')
        if self.transform:
            sample['input_img2_Gray'] = self.transform(sample['input_img2_Gray'])

        sample['input_img2_HR'] = frame
        if self.transform:
            sample['input_img2_HR'] = self.transform(sample['input_img2_HR'])

        if self.require_seqid:
            seq_id = idx
            return sample, seq_id
        else:
            return sample, idx

if __name__ == "__main__":
    data_path_corrupted = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences_noise/'
    data_path_MDSR = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences_upsampled_MDSR/'
    data_path_clean = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sequences/'
    data_list_file = '/fileserver/haitian/Fall2018_Multi_warp/dataset/vimeo_septuplet/sep_trainlist.txt'
    # composed = transforms.Compose([transforms.RandomCrop((128,128)),
    #                                 transforms.ToTensor()])

    composed = transforms.Compose([transforms.ToTensor()])
    dataset = VimeoDataset(data_path_corrupted, data_path_clean, data_path_MDSR, data_list_file, frame_window_size=2,
                           transform=composed)

    #### test pytorch dataset
    # print(len(dataset))

    # fig = plt.figure()
    # plt.axis('off')
    # plt.ioff()
    # im = plt.imshow(np.zeros((dataset.H, dataset.W, 3)), vmin=0, vmax=1)

    # for i in range(len(dataset)-1, 0, -1):
    #     sample = dataset[i]
    #     for t in sample:
    #         print(t, sample[t].size())
    #         im.set_data(sample[t].numpy().transpose(1,2,0))
    #         plt.pause(0.1)
    # exit()

    #### test dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6)
    print(len(dataset), len(dataloader))

    import cPickle as pickle
    import os

    img_name = ['img1_LR', 'img1_SR', 'img1_HR', 'img2_HR']
    for i_batch, sample_batched in enumerate(dataloader):
        # print(i_batch, sample_batched['gt'].size())
        # visualization
        images_batch = sample_batched['input_img1_LR']
        batch_size = images_batch.size()[0]
        im_size = images_batch.size()[1:]

        print(i_batch)
        save_dir = './vimeo_sr/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        f = open(save_dir + str(i_batch + 1), 'wb')
        pickle.dump(sample_batched, f)
        f.close()

        # print(batch_size, im_size)
        # grid = utils.make_grid(images_batch, nrow=2)
        # plt.imshow(grid.numpy().transpose(1,2,0))
        # plt.show()

        # observe 4th batch and stop.
        # if i_batch == 3:
        #     plt.figure()
        #     show_landmarks_batch(sample_batched)
        #     plt.axis('off')
        #     plt.ioff()
        #     plt.show()
        #     break
