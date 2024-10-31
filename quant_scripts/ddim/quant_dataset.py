import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

def get_model_input_time(t_continuous, noise_schedule):
    """
    Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
    For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
    For continuous-time DPMs, we just use `t_continuous`.
    """
    return (t_continuous - 1. / noise_schedule.total_N) * 1000.

class DiffusionInputDataset(Dataset):
    def __init__(self, data_path):
        data_list = torch.load(data_path, map_location='cpu') ## its a list of tuples of tensors
        self.xt_list = []
        self.t_list = []
        self.y_list = []
        ## datalist[i][0].shape (B,4,32,32), flat B dimension
        for i in range(len(data_list)):
            for b in range(len(data_list[i][0])):
                self.xt_list.append(data_list[i][0][b])
                self.t_list.append(data_list[i][1][b])
                self.y_list.append(data_list[i][2][b])

    def __len__(self):
        return len(self.xt_list)
    
    def __getitem__(self, idx):
        return self.xt_list[idx], self.t_list[idx], self.y_list[idx]

class lsunInputDataset(Dataset):
    def __init__(self, data_path):
        data_list = torch.load(data_path, map_location='cpu') ## its a list of tuples of tensors
        self.xt_list = []
        self.t_list = []
        self.y_list = []
        ## datalist[i][0].shape (B,4,32,32), flat B dimension
        for i in range(len(data_list)):
            for b in range(len(data_list[i][0])):
                self.xt_list.append(data_list[i][0][b])
                self.t_list.append(data_list[i][1][b])
                # self.y_list.append(data_list[i][2][b]) ## its None

    def __len__(self):
        return len(self.xt_list)
    
    def __getitem__(self, idx):
        return self.xt_list[idx], self.t_list[idx]

class IntermediateInputDataset(Dataset):
    def __init__(self, data_path):
        data_list = torch.load(data_path, map_location='cpu') ## its a list of tuples of tensors
        # print(data_list.keys())
        # print(data_list)
        xs = data_list['x']
        ts = data_list['t']
        
        self.xt_list = []
        self.t_list = []
        ## datalist[i][0].shape (B,4,32,32), flat B dimension
        for i in range(len(xs)):
            ct = ts[i]
            for b in range(xs[i].shape[0]):
                self.xt_list.append(xs[i][b])
                self.t_list.append(ct)

    def __len__(self):
        return len(self.xt_list)
    
    def __getitem__(self, idx):
        return self.xt_list[idx], self.t_list[idx]
    
from ddim.models.diffusion import NoiseScheduleVP

class SolverIntermediateInputDataset(Dataset):
    def __init__(self, data_path, diffusion):
        data_list = torch.load(data_path, map_location='cpu') ## its a list of tuples of tensors
        # print(data_list.keys())
        # print(data_list)
        xs = data_list['x']
        ts = data_list['t']

        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=diffusion.betas)
        ts = [get_model_input_time(ti, noise_schedule) for ti in ts]
        
        self.xt_list = []
        self.t_list = []
        ## datalist[i][0].shape (B,4,32,32), flat B dimension
        for i in range(len(xs)):
            ct = ts[i]
            for b in range(xs[i].shape[0]):
                self.xt_list.append(xs[i][b])
                self.t_list.append(ct)

    def __len__(self):
        return len(self.xt_list)
    
    def __getitem__(self, idx):
        return self.xt_list[idx], self.t_list[idx]

def get_train_samples(train_loader, num_samples, dataset_type='imagenet'):
    image_data, t_data = [], [] 
    if dataset_type == 'imagenet':
        for (image, t) in train_loader:
            image_data.append(image)
            t_data.append(t)
            if len(image_data) >= num_samples:
                break
                
        return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples]
    
    elif dataset_type == 'lsun':
        for (image, t) in train_loader:
            image_data.append(image)
            t_data.append(t)
            if len(image_data) >= num_samples:
                break
                
        return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples]

def get_calibration_set(data_path, dataset_type='imagenet'):
    # dataset = torch.load(data_path, map_location='cpu')
    # dataset = lsunInputDataset(data_path)
    #data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    if dataset_type == 'imagenet':
        dataset = DiffusionInputDataset(data_path)
        # dataset = torch.load(data_path, map_location='cpu')
        data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
        cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024, dataset_type=dataset_type)
        return cali_images, cali_t, cali_y
    elif dataset_type == 'lsun':
        dataset = lsunInputDataset(data_path)
        data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
        cali_images, cali_t = get_train_samples(data_loader, num_samples=1024, dataset_type=dataset_type)
        return cali_images, cali_t