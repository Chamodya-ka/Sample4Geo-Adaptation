import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
 
class CVUSADatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 fov_phase_seed=1,
                 fov_90=False,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.transforms_query = transforms_query           # ground
        self.transforms_reference = transforms_reference   # satellite
        
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None) # for debug runs with smaller dataset
        
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        self.fov_90 = fov_90
        self.fov_phase_seed = fov_phase_seed
        self.idx2pair = dict()
        train_ids_list = list()
        
        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)
            
        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)

    # def set_fov_phase_seed(self, seed: int):
    #     """Update the crop seed before each similarity-sampling pass."""
    #     self.fov_phase_seed = int(seed)
    def set_epoch(self, epoch: int):
        """Update the epoch for debugging."""
        self.fov_phase_seed = int(epoch)


    def __getitem__(self, index):

        idx, sat, ground = self.idx2pair[self.samples[index]]
        
        # load query -> ground image
        query_img = cv2.imread(f'{self.data_folder}/{ground}')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/{sat}')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Deterministic 90-degree FoV crop.
        # Must happen BEFORE the flip so that the crop region on the original
        # panorama is identical to what CVUSADatasetEval produces for the same
        # idx and fov_phase_seed.  Flipping after the crop is fine for
        # augmentation – it just mirrors the already-cropped patch.
        if self.fov_90 and self.fov_phase_seed is not None:
            h, w = query_img.shape[:2]
            crop_w = w // 4  # 90 / 360 = 0.25
            rng = np.random.default_rng(self.fov_phase_seed * 100003 + idx)
            start = int(rng.integers(0, w - crop_w + 1))
            query_img = query_img[:, start:start + crop_w, :]
            # print(f"[TRAIN]  idx={idx:6d}  fov_phase_seed={self.fov_phase_seed}  start={start}")

        # Flip simultaneously query and reference (after crop so the crop
        # region is consistent with eval / sim-sampling).
        # +1 offset keeps this rng independent from the crop rng (same base seed
        # would produce correlated draws).
        flip_rng = np.random.default_rng(self.fov_phase_seed * 100003 + idx + 1)
        if float(flip_rng.random()) < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']
                
        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2)) 
            
            # use roll for ground view if rotate sat view
            c, h, w = query_img.shape
            shifts = - w//4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)  
                   
            
        label = torch.tensor(idx, dtype=torch.long)  
        
        # denormalize and save samples to visualize 
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        inv_mean = torch.tensor(mean).view(3, 1, 1)
        inv_std = torch.tensor(std).view(3, 1, 1)

        unnormalized_query = (query_img * inv_std) + inv_mean
        unnormalized_reference = (reference_img * inv_std) + inv_mean

        if index < 10:
            cv2.imwrite(f"./debug/{index}_train_query.jpg", cv2.cvtColor(unnormalized_query.permute(1, 2, 0).numpy() * 255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"./debug/{index}_train_reference.jpg", cv2.cvtColor(unnormalized_reference.permute(1, 2, 0).numpy() * 255, cv2.COLOR_RGB2BGR)) 

        return query_img, reference_img, label
    
    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            idx_pool = copy.deepcopy(self.train_ids)
        
            neighbour_split = neighbour_select // 2
            
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)
                
            # Shuffle pairs order
            random.shuffle(idx_pool)
           
            # Lookup if already used in epoch
            idx_epoch = set()   
            idx_batch = set()
     
            # buckets
            batches = []
            current_batch = []
            
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()
    
            while True:
                
                pbar.update()
                
                if len(idx_pool) > 0:
                    idx = idx_pool.pop(0)

                    
                    if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:
                    
                        idx_batch.add(idx)
                        current_batch.append(idx)
                        idx_epoch.add(idx)
                        break_counter = 0
                      
                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = similarity_pool[idx][:neighbour_range]
                            
                            near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                            
                            far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                            
                            random.shuffle(far_neighbours)
                            
                            far_neighbours = far_neighbours[:neighbour_split]
                            
                            near_similarity_select = near_neighbours + far_neighbours
                            
                            for idx_near in near_similarity_select:
                           
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                                
                                # check if idx not already in batch or epoch
                                # also guard against neighbours from sim_dict that are
                                # outside the loaded subset (e.g. during debug runs with nrows)
                                if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near and idx_near in self.idx2pair:
                            
                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0
                                    
                    else:
                        # if idx fits not in batch and is not already used in epoch -> back to pool
                        if idx not in idx_batch and idx not in idx_epoch:
                            idx_pool.append(idx)
                            
                        break_counter += 1
                        
                    if break_counter >= 1024:
                        break
                   
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []

            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            print("idx_pool:", len(idx_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))  

            
       
class CVUSADatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 fov_90=False,
                 fov_phase_seed=0,
                 epoch=0,# just for debugging
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        # fov_phase_seed: when set to an int, applies a deterministic per-image
        # 90-degree horizontal crop in __getitem__ (worker-safe, no global state).
        # Call set_fov_phase_seed(epoch) before each calc_sim run to rotate the
        # crop positions while keeping them consistent across the whole dataset.
        self.fov_phase_seed = fov_phase_seed
        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None) # for debug runs with smaller dataset
        else:
            if fov_90:
                self.df = pd.read_csv(f'{data_folder}/splits/val-19zl-cropped-one-to-one.csv', header=None)
            else:
                self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)
        
        self.df = self.df.rename(columns={0:"sat", 1:"ground", 2:"ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
    
        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values
            
        elif self.img_type == "query":
            self.images = self.df.ground.values
            self.label = self.df.idx.values 
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
    
    def set_epoch(self, epoch: int):
        """Update the epoch for debugging."""
        self.fov_phase_seed = int(epoch)

    # def set_fov_phase_seed(self, seed: int):
    #     """Update the crop seed before each similarity-sampling pass."""
        

    def __getitem__(self, index):
        
        img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Deterministic 90-degree FoV crop for similarity sampling.
        # Uses per-index seeding so behaviour is identical across all DataLoader
        # workers without touching the global random state.
        if self.split == "train" and self.img_type == "query" and self.fov_phase_seed is not None:
            h, w = img.shape[:2]
            crop_w = w // 4  # 90 / 360 = 0.25
            rng = np.random.default_rng(self.fov_phase_seed * 100003 + int(self.label[index]))
            start = int(rng.integers(0, w - crop_w + 1))
            img = img[:, start:start + crop_w, :]
            # print(f"[EVAL ]  idx={int(self.label[index]):6d}  fov_phase_seed={self.fov_phase_seed}  start={start}")
        # if self.img_type == "query":
        #     print("print before save query")
        #     cv2.imwrite(f"./debug/{index}_{self.split}_{self.img_type}.jpg",img)
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)

            





