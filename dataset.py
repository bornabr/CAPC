import glob
import torch
from torch.utils.data import Dataset
import numpy as np


def UT_HAR_dataset(root_dir, portion=None):
	"""Reads UT_HAR dataset and returns WiFi data as tensors.

	Args:
		root_dir (string): Root directory containing UT_HAR data and label files.

	Returns:
		dict: Dictionary containing WiFi data as tensors.
	"""

	WiFi_data = {}
	data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
	label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')

	# Process data files
	for data_dir in data_list:
		data_name = data_dir.split('/')[-1].split('.')[0]
		with open(data_dir, 'rb') as f:
			data = np.load(f)
			data = data.reshape(len(data),1,250,90)
			data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
		WiFi_data[data_name] = torch.Tensor(data_norm)
	
	# Process label files
	for label_dir in label_list:
		label_name = label_dir.split('/')[-1].split('.')[0]
		with open(label_dir, 'rb') as f:
			label = np.load(f)
		WiFi_data[label_name] = torch.Tensor(label).to(torch.int64)
	
	if portion is not None:
		train_x = np.load(f'{root_dir}/UT_HAR/data/unsupervised_{portion}_X.npy')
		train_y = np.load(f'{root_dir}/UT_HAR/label/unsupervised_{portion}_y.npy')
		WiFi_data['X_train'] = torch.Tensor(train_x)
		WiFi_data['y_train'] = torch.Tensor(train_y).to(torch.int64)

	remaining_train_x = np.load(f'{root_dir}/UT_HAR/data/remaining_train_X.npy')
	remaining_train_y = np.load(f'{root_dir}/UT_HAR/label/remaining_train_y.npy')

	WiFi_data['remaining_train_X'] = torch.Tensor(remaining_train_x)
	WiFi_data['remaining_train_y'] = torch.Tensor(remaining_train_y).to(torch.int64)

	# Shape 1 x 250 (Time) x 90 (antenna x subcarrier)
	return WiFi_data

class SignFiDataset(Dataset):
	def __init__(self, root_dir, type, env, link='all', mode='single', portion=12, return_remaining=False):
		self.root_dir = root_dir
		self.env = env
		self.link = link
		self.mode = mode

		self.csid = None
		self.csiu = None
		self.csi = None
		self.label = None

		if link != 'all' and mode == 'dual':
			raise ValueError('dual mode only supports all link')

		if env == 'lab_same' and type == 'train':
			portion = 6 if portion is None else portion
			env = 'lab'
			if return_remaining:
				if link == 'dl':
					self.csid = np.load(self.root_dir + f'reduced(in_env)_{env}_csid_{type}_remaining.npy').transpose(3, 1, 2, 0)
				if link == 'ul':
					self.csiu = np.load(self.root_dir + f'reduced(in_env)_{env}_csiu_{type}_remaining.npy').transpose(3, 1, 2, 0)
				if link == 'all':
					self.csid = np.load(self.root_dir + f'reduced(in_env)_{env}_csid_{type}_remaining.npy').transpose(3, 1, 2, 0)
					self.csiu = np.load(self.root_dir + f'reduced(in_env)_{env}_csiu_{type}_remaining.npy').transpose(3, 1, 2, 0)
				self.label = np.load(self.root_dir + f'reduced(in_env)_{env}_y_{type}_remaining.npy')
				if self.mode == 'single':
					self.csi = np.concatenate((self.csid, self.csiu), axis=3)
					self.label = np.concatenate((self.label, self.label), axis=0)
			else:
				if link == 'dl':
					self.csid = np.load(self.root_dir + f'reduced(in_env)_{env}_csid_{type}_{portion}.npy').transpose(3, 1, 2, 0)
				if link == 'ul':	
					self.csiu = np.load(self.root_dir + f'reduced(in_env)_{env}_csiu_{type}_{portion}.npy').transpose(3, 1, 2, 0)
				if link == 'all':
					self.csid = np.load(self.root_dir + f'reduced(in_env)_{env}_csid_{type}_{portion}.npy').transpose(3, 1, 2, 0)
					self.csiu = np.load(self.root_dir + f'reduced(in_env)_{env}_csiu_{type}_{portion}.npy').transpose(3, 1, 2, 0)
				self.label = np.load(self.root_dir + f'reduced(in_env)_{env}_y_{type}_{portion}.npy')
				if self.mode == 'single':
					self.csi = np.concatenate((self.csid, self.csiu), axis=3)
					self.label = np.concatenate((self.label, self.label), axis=0)
		else:		
			env = 'lab' if env == 'lab_same' else env		
			if portion is not None and type == 'train' and link == 'all' and mode == 'single':
				self.csi = np.load(self.root_dir + f'reduced_{env}_csi_{type}_{portion}.npy')
				self.label = np.load(self.root_dir + f'reduced_{env}_y_{type}_{portion}.npy')
			else:
				if link == 'all':
						self.csid = np.load(self.root_dir + f'{env}_csid_{type}.npy').transpose(2, 1, 0, 3)
						self.csiu = np.load(self.root_dir + f'{env}_csiu_{type}.npy').transpose(2, 1, 0, 3)
						self.label = np.load(self.root_dir + f'{env}_y_{type}.npy') - 1
						if self.mode == 'single':
							self.csi = np.concatenate((self.csid, self.csiu), axis=3)
							self.label = np.concatenate((self.label, self.label), axis=0)
				elif link == 'dl':
					self.csid = np.load(self.root_dir + f'{env}_csid_{type}.npy').transpose(2, 1, 0, 3)
					self.label = np.load(self.root_dir + f'{env}_y_{type}.npy') - 1
				elif link == 'ul':
					self.csiu = np.load(self.root_dir + f'{env}_csiu_{type}.npy').transpose(2, 1, 0, 3)
					self.label = np.load(self.root_dir + f'{env}_y_{type}.npy') - 1
				else:
					raise ValueError('Invalid link type')

				# Get Amplitude
				if self.csid is not None:
					self.csid = np.abs(self.csid)
				if self.csiu is not None:
					self.csiu = np.abs(self.csiu)
				if self.csi is not None:
					self.csi = np.abs(self.csi)

				# Normalize
				if self.csid is not None:
					self.csid = (self.csid - np.min(self.csid)) / (np.max(self.csid) - np.min(self.csid))
				if self.csiu is not None:
					self.csiu = (self.csiu - np.min(self.csiu)) / (np.max(self.csiu) - np.min(self.csiu))
				if self.csi is not None:
					self.csi = (self.csi - np.min(self.csi)) / (np.max(self.csi) - np.min(self.csi))



	def __len__(self):
		if self.mode == 'single':
			return self.csi.shape[3]
		elif self.mode == 'dual':
			return self.csid.shape[3]
		else:
			raise ValueError('Invalid mode type')
	
	def __getitem__(self, idx):
		if self.mode == 'single':
			return torch.DoubleTensor(self.csi[:,:,:,idx]), self.label[idx].astype('int64')
		elif self.mode == 'dual':
			return torch.DoubleTensor(self.csid[:,:,:,idx]), torch.DoubleTensor(self.csiu[:,:,:,idx]), self.label[idx].astype('int64')
		else:
			raise ValueError('Invalid mode type')

def create_loader_from_dataset(train_set, val_set, test_set, batch_size, num_workers, mode):
	if mode == 'train_data':
		unsupervised_train_dataset = train_set
	else:
		if val_set:
			unsupervised_train_dataset = torch.utils.data.ConcatDataset([train_set, val_set, test_set])
		else:
			unsupervised_train_dataset = torch.utils.data.ConcatDataset([train_set, test_set])
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers)
	if val_set:
		val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	unsupervised_train_loader = torch.utils.data.DataLoader(unsupervised_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
	if val_set:
		return train_loader, val_loader, test_loader, unsupervised_train_loader
	else:
		return train_loader, None, test_loader, unsupervised_train_loader


def load_UT_HAR_dataset(root, batch_size, num_workers, mode, portion=None):
	data = UT_HAR_dataset(root, portion)
	unsupervised_train_set = torch.utils.data.TensorDataset(data['remaining_train_X'], data['remaining_train_y'])
	train_set = torch.utils.data.TensorDataset(data['X_train'], data['y_train'])
	val_set = torch.utils.data.TensorDataset(data['X_val'], data['y_val'])
	test_set = torch.utils.data.TensorDataset(data['X_test'], data['y_test'])

	train_loader, val_loader, test_loader, _ = create_loader_from_dataset(train_set, val_set, test_set, batch_size, num_workers, mode)
	unsupervised_train_loader = torch.utils.data.DataLoader(unsupervised_train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

	return train_loader, val_loader, test_loader, unsupervised_train_loader

def load_SignFi_data(root, name, batch_size, num_workers, mode, signfi_env, signfi_link, signfi_mode, portion=12):
	if signfi_env == 'lab_same':
		train_dataset = SignFiDataset(root, 'train', signfi_env, signfi_link, signfi_mode, portion=portion, return_remaining=False)
		unsupervised_train_dataset = SignFiDataset(root, 'train', signfi_env, signfi_link, signfi_mode, return_remaining=True)
		val_dataset = SignFiDataset(root, 'val', signfi_env, signfi_link, signfi_mode, portion=portion)
		test_dataset = SignFiDataset(root, 'test', signfi_env, signfi_link, signfi_mode, portion=portion)

		unsupervised_train_loader = torch.utils.data.DataLoader(unsupervised_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
		
		train_loader, val_loader, test_loader, _ = create_loader_from_dataset(train_dataset, val_dataset, test_dataset, batch_size, num_workers, mode)

		return train_loader, val_loader, test_loader, unsupervised_train_loader
	else:
		train_dataset = SignFiDataset(root, 'train', signfi_env, signfi_link, signfi_mode, portion=portion)
		val_dataset = SignFiDataset(root, 'val', signfi_env, signfi_link, signfi_mode, portion=portion)
		test_dataset = SignFiDataset(root, 'test', signfi_env, signfi_link, signfi_mode, portion=portion)

		return create_loader_from_dataset(train_dataset, val_dataset, test_dataset, batch_size, num_workers, mode)

def data_loader(cfg, num_workers=20, validation_split=0.2):
	root = cfg['root_dir']
	batch_size = cfg['batch_size']
	mode = cfg['mode'] if 'mode' in cfg else None

	if cfg['name'] == 'UT_HAR':
		return load_UT_HAR_dataset(root, batch_size, num_workers, mode, portion=cfg['portion'])
	
	if cfg['type'] == 'SignFi':
		portion = cfg['portion'] if 'portion' in cfg else None
		return load_SignFi_data(root, cfg['name'], batch_size, num_workers, mode, cfg['SignFi_env'], cfg['SignFi_link'], cfg['SignFi_mode'], portion)

	raise ValueError('Invalid dataset type')
