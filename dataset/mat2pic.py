# encoding: utf-8

import os

import cv2
# import h5py
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset, TensorDataset

from utils import project_path

DATAPATH = 'PATH/TO/YOUR/DATA'
PathDict = {}


class LayoutDataset(Dataset):

    def __init__(self, root, transform_name, resize_shape=None, position_type="auto", with_k=False, with_a=False,
                 max_data=-1, random_seed=-1, random_point_num=10):
        assert root or random_seed >= 0 and max_data > 0
        self.root = root
        self.random_seed = random_seed
        self.random_point_num = random_point_num
        self.position_type = position_type

        self.init_mat(max_data)

        assert self.position_type in ["points", "matrix", "xylist", "auto"]
        if self.position_type == "auto":
            mat = self.mats[0]
            datadict = scio.loadmat(os.path.join(self.root, mat))
            if 'list' in datadict and datadict['list'].shape[0] == 1:
                self.position_type = "points"
            elif 'f' in datadict and datadict['f'].shape == (200, 200):
                self.position_type = "matrix"
            elif 'x' in datadict and datadict['x'].shape[0] == 1 and datadict['x'].shape[1] % 2 == 0:
                self.position_type = "xylist"
            else:
                raise NotImplementedError
            print("Detected Position Type:", self.position_type)

        self.resize_shape = resize_shape
        self.transform_func = TransFunc(transform_name, self.position_type)

    def init_mat(self, max_data):
        if self.random_seed >= 0 and max_data > 0:
            self.mats = [str(x) for x in range(self.random_seed, self.random_seed + max_data)]
            if self.position_type == "auto":
                self.position_type = "points"
        elif self.root:
            if self.root in PathDict:
                self.root = PathDict[self.root]
            elif not os.path.exists(self.root):
                if self.root in os.listdir(DATAPATH):
                    self.root = os.path.join(DATAPATH, self.root)
                else:
                    self.root = os.path.join(project_path, self.root)
            assert os.path.exists(self.root), "MAT2PIC ERROR, NO SUCH DIR: {}".format(self.root)
            try:
                self.mats = sorted(os.listdir(self.root), key=lambda x: int(x[:-4]))
            except ValueError:
                self.mats = sorted(os.listdir(self.root))
                print("WARNING: mats should be named like 1.mat.")
            max_data = min(max_data, len(self.mats)) if max_data > 0 else len(self.mats)
            self.mats = self.mats[:max_data]
        else:
            raise ValueError("You must specify data root or random seed.")

    def __len__(self):
        return len(self.mats)

    def __getitem__(self, item):
        mat = self.mats[item]
        if self.random_seed >= 0:
            np.random.seed(int(mat))
            if self.position_type == "points":
                location = np.random.choice(range(100), self.random_point_num, replace=False) + 1
            elif self.position_type == "xylist":
                pointlist = list(range(181 * 181))
                location_x, location_y = [], []
                for p_index in range(self.random_point_num):
                    if len(pointlist) == 0:
                        print("WARNING: Only {} points generated.".format(p_index))
                        break
                    p_point = np.random.choice(pointlist, 1)[0]
                    p_point_x, p_point_y = p_point // 181, p_point % 181
                    location_x.append(p_point_x)
                    location_y.append(p_point_y)
                    to_remove = set(ii * 181 + jj
                                    for ii in range(max(p_point_x - 19, 0), min(p_point_x + 20, 181))
                                    for jj in range(max(p_point_y - 19, 0), min(p_point_y + 20, 181)))
                    pointlist = [pp_point for pp_point in pointlist if pp_point not in to_remove]
                location = np.array(location_x + location_y)
                location += 1
            elif self.position_type == "matrix":
                location = np.random.rand(200, 200)
            # heat_map = (datadict['u']-290)/100
            heat_map = np.random.rand(200, 200)
        else:
            datadict = scio.loadmat(os.path.join(self.root, mat))
            if self.position_type == "points":
                location = datadict['list'][0]
            elif self.position_type == "xylist":
                location = datadict['x'][0]
            elif self.position_type == "matrix":
                location = datadict['f']
            if 'u' in datadict:
                heat_map = (datadict['u'] - 290) / 100
                # heat_map = (datadict['u']-298)
            elif 'uu' in datadict:
                heat_map = (datadict['uu'] - 298) / 10
            else:
                raise ValueError("Cannot find HeatMap")
        return self.transform_func(location, heat_map, self.resize_shape)


class OneDataset(LayoutDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_mat(self, max_data):
        if not os.path.exists(self.root):
            if os.path.exists(os.path.join(DATAPATH, self.root)):
                self.root = os.path.join(DATAPATH, self.root)
            elif os.path.exists(os.path.join(project_path, self.root)):
                self.root = os.path.join(project_path, self.root)
            else:
                raise ValueError("Wrong Data Path")
        self.root, self.mats = os.path.split(self.root)
        self.mats = [self.mats]


class TransFunc(object):
    def __init__(self, transform_func, position_type="points"):
        if isinstance(transform_func, str):
            self.transform_func = eval("self.trans_" + transform_func)
        else:
            raise NotImplementedError

        self.preprocess_dict = {
            "points": self.layout2map,
            "xylist": self.layout2map_xy,
            "xyxy": self.layout2map_xyxy,
            "matrix": None,
        }

        assert position_type in ["points", "xylist", "xyxy", "matrix"]

        self.preprocess = self.preprocess_dict[position_type]

    @staticmethod
    def layout2map(location):
        layout_map = np.zeros((200, 200))
        location -= 1
        for i in location:
            layout_map[(i % 10) * 20:(i % 10) * 20 + 20, i // 10 * 20:i // 10 * 20 + 20] = np.ones((20, 20))
        return layout_map

    @staticmethod
    def layout2map_xy(location):
        layout_map = np.zeros((200, 200))
        location -= 1
        assert len(location) % 2 == 0
        half = len(location) // 2
        for x, y in zip(location[:half], location[half:]):
            layout_map[x:x + 20, y:y + 20] = np.ones((20, 20))
        return layout_map

    @staticmethod
    def layout2map_xyxy(location):
        layout_map = np.zeros((200, 200))
        location -= 1
        for x, y in location:
            layout_map[x * 20:x * 20 + 20, y * 20:y * 20 + 20] = np.ones((20, 20))
        return layout_map

    def trans_vstack(self, layout_map, heat, resize_shape):
        if self.preprocess is not None:
            layout_map = self.preprocess(layout_map)
        res = np.vstack((layout_map, heat))
        if resize_shape:
            res = cv2.resize(res, resize_shape)
        res = np.expand_dims(res, 0)
        return torch.from_numpy(res.astype(np.float32))

    def trans_channels(self, layout_map, heat_map, resize_shape):
        if self.preprocess is not None:
            layout_map = self.preprocess(layout_map)
        if resize_shape:
            res = np.array([cv2.resize(layout_map, resize_shape), cv2.resize(heat_map, resize_shape)])
        else:
            res = np.array([layout_map, heat_map])
        return torch.from_numpy(res.astype(np.float32))

    def trans_separate(self, layout_map, heat_map, resize_shape):
        if self.preprocess is not None:
            layout_map = self.preprocess(layout_map)
        if resize_shape:
            layout_map = cv2.resize(layout_map, resize_shape)
            heat_map = cv2.resize(heat_map, resize_shape)
        layout_map = np.expand_dims(layout_map, 0)
        heat_map = np.expand_dims(heat_map, 0)
        return torch.from_numpy(layout_map.astype(np.float32)), torch.from_numpy(heat_map.astype(np.float32))

    def trans_sar(self, layout_map, heat_map, resize_shape):
        layout_map = np.array(layout_map) - 1
        if resize_shape:
            heat_map = cv2.resize(heat_map, resize_shape)
        heat_map = np.expand_dims(heat_map, 0)
        return torch.from_numpy(layout_map).long(), torch.from_numpy(heat_map.astype(np.float32))

    def __call__(self, *args, **kwargs):
        return self.transform_func(*args, **kwargs)


# class HDF5Dataset(TensorDataset):
#     def __init__(self, file):
#         f = h5py.File(file, "r")
#         layout_map = torch.FloatTensor(f["input"]) / -1e4
#         layout_map = layout_map.unsqueeze(-1).repeat(1, 1, 1, 1, 4).view(1, 1, 50, 200)
#         layout_map = layout_map.unsqueeze(-2).repeat(1, 1, 1, 4, 1).view(1, 1, 200, 200)
#         heat_map = torch.FloatTensor(f["output"]) - 298
#         data_tuple = (layout_map, heat_map)
#         super().__init__(*data_tuple)


if __name__ == "__main__":
    print("Testing Layout Dataset ...")
    dataset = LayoutDataset("test", "channels", (200, 200), max_data=100, random_seed=1, position_type="xylist")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for it, images in enumerate(data_loader):
        print(images.shape)
        print(torch.max(images), torch.min(images))
        break
    # print("Testing HDF5 Dataset ...")
    # dataset = HDF5Dataset()
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    # for it, (layout, heat) in enumerate(data_loader):
    #     print(layout.size(), heat.size())
    #     print(torch.max(layout), torch.min(layout))
    #     break
    print("Testing One Dataset ...")
    dataset = OneDataset("test/1.mat", "channels", (200, 200), position_type="auto")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for it, images in enumerate(data_loader):
        print(images.shape)
        print(torch.max(images), torch.min(images))
        break
    print("Test Completed Successfully!")
