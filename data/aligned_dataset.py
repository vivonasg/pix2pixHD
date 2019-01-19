### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import pdb
import numpy as np
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        
        ### input A (label maps)
        dir_A = 'normals_images' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot,dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        
        ### input B (real images)
        if opt.isTrain:
            dir_B = 'seam_images' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot,dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)                     
        
        
    def __getitem__(self, index):        
        ### input A (label maps)
        
        A_path = self.A_paths[index]   
        A=Image.open(A_path)
        #A=2*(A/255) -1

        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
            if 'seams' in self.opt.resize_or_crop:
                B_tensor=2*(A_tensor/255)-1 #scale between -1 and 1
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        
        
        
        if self.opt.isTrain:
            
            B_path = self.B_paths[index]   
            B = Image.open(B_path)
            
            if 'seams' in self.opt.resize_or_crop:
                B = Image.fromarray(self.get_one_hot_mask(np.array(A), np.array(B)).astype('uint8'), 'RGB')
            
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)
            
            if 'seams' in self.opt.resize_or_crop:
                B_tensor=2*(B_tensor)-1 #go from -1 to 1

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict
    def get_one_hot_mask(self, normals_map, seam_map):
        rows,cols=normals_map.shape[0],normals_map.shape[1]
        num_class=3
        def binarylab(labels, num_class, rows, cols):
            x = np.zeros([rows,cols,num_class])
            for i in range(rows):
                for j in range(cols):
                    x[i,j,int(labels[i][j])]=1
            return x
        black = [0, 0, 0] #Threshold to define background
        bl = 0
        #Images with 0s and 1s where 1 define Silhouette or Seam
        silhouette = np.any(normals_map > black, axis=2).astype(np.uint8)*1  
        seams = np.where(seam_map > bl, 1, 0).astype(np.uint8)*1
        dumy_mask = silhouette + seams #1 Channel Mask
        one_hot_mask = binarylab(dumy_mask, num_class, rows, cols)
        return one_hot_mask
    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'