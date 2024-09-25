import math
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import random
from torch.optim.lr_scheduler import LambdaLR
import wandb

from modules import projector, RecurrentEncoder

def KL(probs1, probs2, args):
	kl = (probs1 * (probs1 + args.model['EPS']).log() - probs1 * (probs2 + args.model['EPS']).log()).sum(dim=1)
	kl = kl.mean()
	return kl

def CE(probs1, probs2, args):
	ce = - (probs1 * (probs2 + args.model['EPS']).log()).sum(dim=1)
	ce = ce.mean()
	return ce

def HE(probs, args): 
	mean = probs.mean(dim=0)
	ent  = - (mean * (mean + args.model['EPS']).log()).sum()
	return ent

def EH(probs, args):
	ent = - (probs * (probs + args.model['EPS']).log()).sum(dim=1)
	mean = ent.mean()
	return mean

def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
	# Normalize each vector by its norm
	output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
	output_net = output_net / (output_net_norm + eps)
	output_net[output_net != output_net] = 0

	target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
	target_net = target_net / (target_net_norm + eps)
	target_net[target_net != target_net] = 0

	# Calculate the cosine similarity
	model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
	target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

	# Scale cosine similarity to 0..1
	model_similarity = (model_similarity + 1.0) / 2.0
	target_similarity = (target_similarity + 1.0) / 2.0

	# Transform them into probabilities
	model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
	target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

	# Calculate the KL-divergence
	loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

	return loss

def off_diagonal(x):
	# return a flattened view of the off-diagonal elements of a square matrix
	n, m = x.shape
	assert n == m
	return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SSLModel(pl.LightningModule):
	def __init__(self, hparams):
		super(SSLModel, self).__init__()
		self.save_hyperparameters(hparams)

		if 'n_hidden_states_nodes_last_layer' not in self.hparams.model:
			self.hparams.model['n_hidden_states_nodes_last_layer'] = self.hparams.model['n_hidden_states_nodes']


		self.encoder = RecurrentEncoder(self.hparams.dataset['type'], self.hparams.model['num_frames'], self.hparams.model['embedding_size'], self.hparams.model['recurrent_block'], self.hparams.model['augmentations'])
		if self.hparams.model['shared_weights']:
			self.encoder_2 = self.encoder
		else:
			self.encoder_2 = RecurrentEncoder(self.hparams.dataset['type'], self.hparams.model['num_frames'], self.hparams.model['embedding_size'], self.hparams.model['recurrent_block'], self.hparams.model['augmentations'])
		

		self.projector = projector(self.hparams.model['n_hidden_states_nodes'], self.hparams.model['n_hidden_states_nodes_last_layer'], self.hparams.model['embedding_size'])
		if self.hparams.model['shared_weights']:
			self.projector_2 = self.projector
		else:
			self.projector_2 = projector(self.hparams.model['n_hidden_states_nodes'], self.hparams.model['n_hidden_states_nodes_last_layer'], self.hparams.model['embedding_size'])

		self.bn = nn.BatchNorm1d(self.hparams.model['n_hidden_states_nodes_last_layer'], affine=False)

		if 'CPC' in self.hparams.model['losses']:
			self.timestep = self.hparams.model['timestep']
			if 'cpc_autoregressive_model' not in self.hparams.model or self.hparams.model['cpc_autoregressive_model'] == 'GRU': 
				self.autoregressive_model = nn.GRU(self.hparams.model['embedding_size'], self.hparams.model['n_hidden_states_nodes_last_layer'], batch_first=True, num_layers=1, bidirectional=False)
			elif self.hparams.model['cpc_autoregressive_model'] == 'LSTM':
				self.autoregressive_model = nn.LSTM(self.hparams.model['embedding_size'], self.hparams.model['n_hidden_states_nodes_last_layer'], batch_first=True, num_layers=1, bidirectional=False)
			elif self.hparams.model['cpc_autoregressive_model'] == 'RNN':
				self.autoregressive_model = nn.RNN(self.hparams.model['embedding_size'], self.hparams.model['n_hidden_states_nodes_last_layer'], batch_first=True, num_layers=1, bidirectional=False)
			self.Wk = nn.ModuleList([nn.Linear(self.hparams.model['n_hidden_states_nodes_last_layer'], self.hparams.model['embedding_size']) for _ in range(self.timestep)])
			self.softmax = nn.Softmax(dim=1)
			self.lsoftmax = nn.LogSoftmax(dim=1)

	def probability_consistency_loss(self, probs1, probs2):
		return 0.5 * (KL(probs1, probs2, self.hparams) + KL(probs2, probs1, self.hparams))

	def mutual_information_loss(self, probs1, probs2):
		loss_eh = 0.5 * (EH(probs1, self.hparams) + EH(probs2, self.hparams))
				
		loss_he = 0.5 * (HE(probs1, self.hparams) + HE(probs2, self.hparams))
				
		return loss_eh - loss_he

	def geometric_consistency_loss(self, f1, f2):
		return 1000*cosine_similarity_loss(f1, f2)
	
	def barlow_twin_loss(self, z1, z2, batch_size):
		c = self.bn(z1).T @ self.bn(z2) / batch_size

		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(c).pow_(2).sum()
		return on_diag + self.hparams.model['lambd'] * off_diag, on_diag, off_diag

	def barlow_twin_ncpc_loss(self, z1, z2, batch_size):
		# print devices
		c = self.bn2(z1).T @ self.bn2(z2) / batch_size

		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(c).pow_(2).sum()
		return on_diag + self.hparams.model['lambd'] * off_diag, on_diag, off_diag

	def invariance_loss(self, z1, z2):
		return self.hparams.model['sim_coeff'] * F.mse_loss(z1, z2)
	
	def variance_loss(self, z1, z2):
		std_z1 = torch.sqrt(z1.var(dim=0) + self.hparams.model['EPS'])
		std_z2 = torch.sqrt(z2.var(dim=0) + self.hparams.model['EPS'])
		return self.hparams.model['std_coeff'] * (torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2)

	def covariance_loss(self, z1, z2, batch_size):
		z1 = z1 - z1.mean(dim=0)
		z2 = z2 - z2.mean(dim=0)
		cov_z1 = (z1.T @ z1) / (batch_size - 1)
		cov_z2 = (z2.T @ z2) / (batch_size - 1)
		cov_loss = off_diagonal(cov_z1).pow_(2).sum().div(self.hparams.model['n_hidden_states_nodes_last_layer']) + off_diagonal(cov_z2).pow_(2).sum().div(self.hparams.model['n_hidden_states_nodes_last_layer'])
		return self.hparams.model['cov_coeff'] * cov_loss

	def SimCLR_loss(self, z1, z2, batch_size):	
		# Normalize the projections
		z1 = F.normalize(z1, dim=1)
		z2 = F.normalize(z2, dim=1)

		# Concatenate the normalized projections
		representations = torch.cat([z1, z2], dim=0)

		# Compute similarity matrix
		similarity_matrix = torch.matmul(representations, representations.T)

		# Create positive mask
		mask = torch.eye(batch_size, dtype=torch.bool)
		mask = mask.repeat(2, 2)
		mask = mask.fill_diagonal_(False)

		# Compute logits
		# logits = similarity_matrix / self.hparams.model['temperature']
		# logits = logits.float()

		# # Compute labels
		# labels = torch.arange(batch_size).type_as(logits)
		# labels = torch.cat([labels + batch_size, labels], dim=0)

		# Compute loss
		# loss = F.cross_entropy(logits[mask].view(2 * batch_size, -1).squeeze(), labels)


		similarity_matrix = similarity_matrix / self.hparams.model['temperature']
		similarity_matrix = similarity_matrix.float()

		sim_1_2 = torch.diag(similarity_matrix, batch_size)
		sim_2_1 = torch.diag(similarity_matrix, -batch_size)

		positive_samples = torch.cat((sim_1_2, sim_2_1), dim=0).view(2 * batch_size, 1)
		negative_samples = similarity_matrix[~mask].view(2 * batch_size, -1)

		labels = torch.zeros(2 * batch_size).type_as(similarity_matrix).long()
		labels[0:batch_size] = 1

		logits = torch.cat((positive_samples, negative_samples), dim=1)
		
		loss = F.cross_entropy(logits, labels,reduction="sum")

		return loss

	def CPC(self, f, batch_size, t_samples=None):
		if t_samples is None:
			t_samples = torch.randint(self.encoder.sequence_length-self.timestep, size=(1,)).long() # randomly pick time stamps.
		
		f = f.view(batch_size, self.encoder.sequence_length, f.shape[-1])
		# encode sequence is N*L*D, where N is batch size, L is sequence length, D is feature dimension (e.g. 128*4*512)
		
		encode_samples = torch.empty((self.timestep, batch_size, f.shape[-1]), dtype=f.dtype, device=f.device) # e.g. 4*128*512

		for i in range(1, self.timestep+1):
			encode_samples[i-1] = f[:, t_samples+i, :].view(batch_size, -1) # e.g. 128*512
		
		forward_seq = f[:, :t_samples+1, :] # e.g. 128*2*512
		output, _ = self.autoregressive_model(forward_seq) # e.g. 128*2*256
		c_t = output[:, t_samples, :].view(batch_size, -1) # e.g. 128*256
		pred = torch.empty((self.timestep, batch_size, f.shape[-1]), dtype=f.dtype, device=f.device)
		
		for i in range(self.timestep):
			linear = self.Wk[i]
			pred[i] = linear(c_t)
		
		return encode_samples, pred, c_t


	def gaussian_noise(self, csi):
		"""
		Adds Gaussian noise to the CSI tensor.

		Args:
		- csi (torch.Tensor): The CSI tensor to be augmented. Expected shape is (dim1, dim2, dim3).

		Returns:
		- torch.Tensor: The augmented CSI tensor.
		"""
		# noise = torch.normal(1, 2, size=csi.shape[1:]).cuda()
		noise = torch.randn(csi.shape[1:]) * self.hparams.model['augmentations']['noise'][1] + self.hparams.model['augmentations']['noise'][0]
		# noise = torch.normal(self.hparams.model['augmentations']['noise'][0], self.hparams.model['augmentations']['noise'][1], size=csi.shape[1:]).cuda()
		perturbed_csi = csi + noise.type_as(csi)
		return perturbed_csi

	def random_flip_tensor(self, csi_tensor):
		"""
		Randomly flips the CSI tensor along specified dimensions.

		Args:
		- csi_tensor (torch.Tensor): The CSI tensor to be augmented. Expected shape is (dim1, dim2, dim3).
		- dimensions_to_flip (list of int): A list of dimensions (0, 1, or 2) along which the tensor might be flipped.

		Returns:
		- torch.Tensor: The possibly flipped CSI tensor.
		"""

		dims = set()
		for dim in self.hparams.model['augmentations']['flip']:
			if type(dim) is int:
				dims.add(dim)
			elif dim in self.hparams.dataset['dimension_maps']:
				dims.add(int(self.hparams.dataset['dimension_maps'][dim]))
			else:
				raise ValueError("Invalid dimension to flip.")
		
		for dim in dims:
			if torch.rand(1) > 0.5:  # 50% chance for flipping
				csi_tensor = torch.flip(csi_tensor, [dim])

		return csi_tensor
	
	def zero_masking(self, tensor):
		"""
		Applies zero masking along a specified dimension of the tensor.

		Args:
		- tensor (torch.Tensor): The input tensor. Can be of any shape.
		- dim (int): The dimension along which the masking should be applied.
		- mask_length (int): The length of the zero mask.

		Returns:
		- torch.Tensor: The tensor with a segment zeroed out along the specified dimension.
		"""
		ratios = self.hparams.model['augmentations']['zero_masking']['ratio']
		dims = self.hparams.model['augmentations']['zero_masking']['dim']

		if isinstance(dims) is not list:
			dims = [dims]
			ratios = [ratios]

		for dim, ratio in zip(dims, ratios):
			dim_int = None

			if type(dim) is not int:
				if dim in self.hparams.dataset['dimension_maps']:
					dim_int = int(self.hparams.dataset['dimension_maps'][dim])
				else:
					raise ValueError("Invalid dimension to zero mask.")
			else:
				dim_int = dim

			mask_length = int(ratio * tensor.size(dim_int))

			# Check if mask length is valid
			if mask_length > tensor.size(dim_int):
				raise ValueError(f"Mask length {mask_length} is greater than tensor dimension {tensor.size(dim_int)} along axis {dim}")

			if dim == 'time':
				# Randomly choose the start index for zero masking
				start_idx = torch.randint(0, tensor.size(dim_int) - mask_length + 1, (1,))

				# Create a mask of ones
				mask = torch.ones_like(tensor)

				# Set the segment to zero in the mask
				if dim_int == 1:
					mask[:, start_idx:start_idx+mask_length, ...] = 0
				elif dim_int == 2:
					mask[:, :, start_idx:start_idx+mask_length, ...] = 0
				elif dim_int == 3:
					mask[:, :, :, start_idx:start_idx+mask_length, ...] = 0
				# Extend for more dimensions if necessary...

				# Apply mask
				tensor = tensor * mask
			else:
				# Raandomly choose the items to be masked
				masked_items = torch.randperm(tensor.size(dim_int))[:mask_length]

				# Create a mask of ones
				mask = torch.ones_like(tensor)

				# Set the items to zero in the mask
				if dim_int == 1:
					mask[:, masked_items, ...] = 0
				elif dim_int == 2:
					mask[:, :, masked_items, ...] = 0
				elif dim_int == 3:
					mask[:, :, :, masked_items, ...] = 0

				# Apply mask
				tensor = tensor * mask

		return tensor

	def time_shift(tensor, dim, shift_amount):
		"""
		Shifts the tensor along a specified dimension.

		Args:
		- tensor (torch.Tensor): The input tensor. Can be of any shape.
		- dim (int): The dimension along which the shift should be applied.
		- shift_amount (int): The amount by which the tensor should be shifted. Can be positive or negative.

		Returns:
		- torch.Tensor: The shifted tensor.
		"""

		if abs(shift_amount) > tensor.size(dim):
			raise ValueError(f"Shift amount {shift_amount} exceeds tensor dimension {tensor.size(dim)} along axis {dim}")

		# Roll tensor values
		shifted_tensor = torch.roll(tensor, shifts=shift_amount, dims=dim)

		return shifted_tensor

	def dual_view_augmentaion(self, batch):
		if 'dual_view' in self.hparams.model['augmentations'] and self.hparams.model['augmentations']['dual_view']:
			tmp1, tmp2, y = batch
			if random.random() > 0.5:
				real_x1 = tmp1
				real_x2 = tmp2
			else:
				real_x1 = tmp2
				real_x2 = tmp1
			x1 = real_x1
			x2 = real_x2
		else:
			x, y = batch
			real_x1 = x
			real_x2 = x
			x1 = real_x1
			x2 = real_x2

		return x1, x2, real_x1, real_x2, y

	def augmentation(self, x):
		if 'flip' in self.hparams.model['augmentations'] and self.hparams.model['augmentations']['flip']:
			x = self.random_flip_tensor(x)
		if 'noise' in self.hparams.model['augmentations'] and self.hparams.model['augmentations']['noise']:
			x = self.gaussian_noise(x)
		return x

	def training_step(self, batch, batch_idx):
		if self.trainer.global_step == 0 and self.hparams.wandb: 
			wandb.define_metric('train_loss', summary='min')

		x1, x2, real_x1, real_x2, y = self.dual_view_augmentaion(batch)

		x1, x2 = self.augmentation(x1), self.augmentation(x2)

		batch_size = y.shape[0]
		tmp_batch_size = batch_size
		f1, tmp_batch_size = self.encoder(x1, view_mode='in_batch', mode='self_supervised')
		f2, tmp_batch_size = self.encoder_2(x2, view_mode='in_batch', mode='self_supervised')
		

		loss_functions = []
		if 'CPC' in self.hparams.model['losses']:
			if len(self.hparams.model['losses']) > 1:
				branches = [f1, f2]
				tmp_batch_size = batch_size
			else:
				branches = [f1]
			
			loss_cpc_array = []
			c_t_array = []

			t_samples = torch.randint(self.encoder.sequence_length-self.timestep, size=(1,)).long() # randomly pick time stamps.
			
			cpc_array = []
			for i, f in enumerate(branches):
				encode_samples, pred, c_t = self.CPC(f, batch_size, t_samples)
				cpc_array.append((encode_samples, pred, c_t))
			
			for index, f in enumerate(branches):
				encode_samples, pred, c_t = cpc_array[index]

				if len(self.hparams.model['losses']) > 1:
					encode_samples_2, pred_2, c_t_2 = cpc_array[1-index]

				nce = 0
				correct = 0

				for i in range(self.timestep):
					# stop gradient for the future sample
					# sample = encode_samples[i].detach()
					sample = encode_samples[i]

					# if len(self.hparams.model['losses']) > 1:
					# 	total = torch.mm(sample, torch.transpose(pred_2[i], 0, 1)) # e.g. 128*128
					# else:
					# 	total = torch.mm(sample, torch.transpose(pred[i], 0, 1))

					total = torch.mm(sample, torch.transpose(pred[i], 0, 1))

					correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), 0), torch.arange(0, batch_size, dtype=f.dtype, device=f.device)))
					nce += -torch.sum(torch.diag(self.lsoftmax(total)))
				loss_cpc = nce / (batch_size * self.timestep)
				accuracy = correct.item() / (batch_size    * self.timestep)
				self.log(f'train_cpc_accuracy_{index}', accuracy)
				self.log(f'train_loss_cpc_{index}', loss_cpc)

				loss_cpc_array.append(loss_cpc)
				c_t_array.append(c_t)

		
			loss_cpc = sum(loss_cpc_array)


			if len(self.hparams.model['losses']) > 1:
				self.log('train_loss_cpc', loss_cpc)
				loss_functions.append(self.hparams.model['cpc_coeff'] * loss_cpc)
				f1, f2 = c_t_array
			else:
				self.log('train_loss', loss_cpc)
				return loss_cpc

		if ('remove_projector' in self.hparams.model and self.hparams.model['remove_projector']) or ('remove_projector' not in self.hparams.model and ('CPC' in self.hparams.model['losses'] or 'NCPC' in self.hparams.model['losses'])):
			z1, z2 = f1, f2
		else:
				z1 = self.projector(f1)
				z2 = self.projector_2(f2)

		p1 = nn.functional.softmax(z1, dim=-1)
		p2 = nn.functional.softmax(z2, dim=-1)
		
		if 'barlow_twin' in self.hparams.model['losses']:
			loss_bt, on_diag, off_diag = self.barlow_twin_loss(z1, z2, batch_size)
			loss_functions.append(loss_bt)
			self.log('train_loss_bt', loss_bt)
			self.log('train_loss_on_diag', on_diag)
			self.log('train_loss_off_diag', off_diag)
		if 'invariance' in self.hparams.model['losses']:
			loss_i = self.invariance_loss(z1, z2)
			loss_functions.append(loss_i)
			self.log('train_loss_i', loss_i)
		if 'variance' in self.hparams.model['losses']:
			loss_v = self.variance_loss(z1, z2)
			loss_functions.append(loss_v)
			self.log('train_loss_v', loss_v)
		if 'covariance' in self.hparams.model['losses']:
			loss_cov = self.covariance_loss(z1, z2, batch_size)
			loss_functions.append(loss_cov)
			self.log('train_loss_cov', loss_cov)
		if 'probability_consistency' in self.hparams.model['losses']:
			loss_p = self.probability_consistency_loss(p1, p2)
			loss_functions.append(loss_p)
			self.log('train_loss_p', loss_p)
		if 'mutual_information' in self.hparams.model['losses']:
			loss_m = self.mutual_information_loss(p1, p2)
			loss_functions.append(loss_m)
			self.log('train_loss_m', loss_m)
		if 'geometric_consistency' in self.hparams.model['losses']:
			loss_g = self.geometric_consistency_loss(f1, f2)
			if 'barlow_twin' in self.hparams.model['losses']:
				loss_g = loss_g * 1000
			loss_functions.append(loss_g)
			self.log('train_loss_g', loss_g)
		if 'SimCLR' in self.hparams.model['losses']:
			loss_s = self.SimCLR_loss(z1, z2, tmp_batch_size)
			loss_functions.append(loss_s)
			self.log('train_loss_s', loss_s)

		loss = sum(loss_functions)

		self.log("train_loss", loss, prog_bar=True)

		return loss

	def configure_optimizers(self):
		
		param_weights = []
		param_biases = []
		for param in self.parameters():
			if param.ndim == 1:
				param_biases.append(param)
			else:
				param_weights.append(param)
		parameters = [{'params': param_weights}, {'params': param_biases}]
		
		optimizer = LARS(parameters, lr=self.hparams.dataset['batch_size'] / 256, weight_decay=self.hparams.model['weight_decay'],
						 weight_decay_filter=True,
						 lars_adaptation_filter=True, logger=self.log)

		def lr_lambda_weights(step):
			# Learning rate scheduler logic
			max_steps = self.hparams.model['epochs'] * self.trainer.num_training_batches
			warmup_steps = 10 * self.trainer.num_training_batches
			base_lr = self.hparams.dataset['batch_size'] / 256
			if step < warmup_steps:
				lr = base_lr * step / warmup_steps
			else:
				step -= warmup_steps
				max_steps -= warmup_steps
				q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
				end_lr = base_lr * 0.001
				lr = base_lr * q + end_lr * (1 - q)
			return lr * self.hparams.model['lr']

		def lr_lambda_biases(step):
			# Learning rate scheduler logic
			max_steps = self.hparams.model['epochs'] * self.trainer.num_training_batches
			warmup_steps = 10 * self.trainer.num_training_batches
			base_lr = self.hparams.dataset['batch_size'] / 256
			if step < warmup_steps:
				# lr = base_lr * step / warmup_steps
				lr = step / warmup_steps
			else:
				step -= warmup_steps
				max_steps -= warmup_steps
				q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
				end_lr = base_lr * 0.001
				# lr = base_lr * q + end_lr * (1 - q)
				lr = q + end_lr * (1 - q)
			return lr * self.hparams.model['lr_biases']

		lr_scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda_weights, lr_lambda_biases])

		return {
			'optimizer': optimizer, 
			'lr_scheduler': {
				'scheduler': lr_scheduler, 
				'interval': 'step'
			}
		}

class LinearClassifierModel(pl.LightningModule):
	def __init__(self, pretrained_encoder, hparams):
		super(LinearClassifierModel, self).__init__()
		self.save_hyperparameters(hparams)

		# Assign the pretrained encoder to the model's encoder
		self.encoder = pretrained_encoder

		# Check if the model is not semi-supervised
		if self.hparams['freeze_encoder']:
			# Freeze all parameters of the encoder
			for param in self.encoder.parameters():
				param.requires_grad = False

		self.linear_seperation = nn.Linear(self.encoder.embedding_size*self.encoder.sequence_length, self.hparams.dataset['num_classes'])

		# Set automatic optimization to False
		self.automatic_optimization = False

		self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams.dataset['num_classes'])
		self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams.dataset['num_classes'])
		self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams.dataset['num_classes'])


	def forward(self, x):
		x, _ = self.encoder(x, view_mode='flat')
		x = self.linear_seperation(x)
		return x
	
	def training_step(self, batch, batch_idx):
		x, y = batch

		y_hat = self(x)

		loss = F.cross_entropy(y_hat, y.squeeze())

		self.train_accuracy.update(y_hat, y.squeeze())

		# Check if we are in semi-supervised mode to decide how many optimizers we have
		if 'semi_supervised' in self.hparams and self.hparams['semi_supervised']:
			# Access optimizers
			optimizer_encoder, optimizer_classifier = self.optimizers()
			
			# Manually zero the gradients
			optimizer_encoder.zero_grad()
			optimizer_classifier.zero_grad()
			
			# Manually perform backward pass for the loss
			self.manual_backward(loss)
			
			# Manually step the optimizers
			optimizer_encoder.step()
			optimizer_classifier.step()
		else:
			# If not semi_supervised, we have only one optimizer
			optimizer = self.optimizers()
			optimizer.zero_grad()
			self.manual_backward(loss)
			optimizer.step()

		self.log("train_loss", loss, on_step=True, prog_bar=True)

		return loss
	
	def validation_step(self, batch, batch_idx):
		if self.trainer.global_step == 0 and self.hparams.wandb: 
			wandb.define_metric('val_acc_epoch', summary='max')
			wandb.define_metric('val_loss', summary='min')
		
		x, y = batch

		y_hat = self(x)

		loss = F.cross_entropy(y_hat, y.squeeze())

		self.val_accuracy.update(y_hat, y.squeeze())

		self.log("val_loss", loss)

		return loss


	def test_step(self, batch, batch_idx):
		x, y = batch

		y_hat = self(x)

		loss = F.cross_entropy(y_hat, y.squeeze())

		self.log("test_loss", loss)

		self.test_accuracy.update(y_hat, y.squeeze())

		return loss

	def on_train_epoch_end(self):
		self.log('train_acc_epoch', self.train_accuracy.compute())
		self.train_accuracy.reset()
		
		# Check if we are in semi-supervised mode to decide how many schedulers we have
		if 'semi_supervised' in self.hparams and self.hparams['semi_supervised']:
			# Access schedulers
			scheduler_encoder, scheduler_classifier = self.lr_schedulers()

			# Step the schedulers
			scheduler_encoder.step()
			scheduler_classifier.step()
		else:
			# If not semi_supervised, we have only one scheduler
			scheduler = self.lr_schedulers()

			# Step the scheduler
			scheduler.step()
	
	
	def on_validation_epoch_end(self):
		self.log('val_acc_epoch', self.val_accuracy)

	def on_test_epoch_end(self):
		self.log('test_acc_epoch', self.test_accuracy.compute())
		self.test_accuracy.reset()
	
	def configure_optimizers(self):
		if self.hparams['semi_supervised']:
			optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=self.hparams.model['lr_encoder'])
			optimizer_classifier = torch.optim.Adam(self.linear_seperation.parameters(), lr=self.hparams.model['lr'])

			scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=self.hparams.model['epochs'] if self.hparams.model['epochs'] is not None else self.hparams.model['steps'])
			scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_classifier, T_max=self.hparams.model['epochs'] if self.hparams.model['epochs'] is not None else self.hparams.model['steps'])

			return [optimizer_encoder, optimizer_classifier], [scheduler_encoder, scheduler_classifier]
		else:
			optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.model['lr'], weight_decay=self.hparams.model['weight_decay'])
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.model['epochs'] if self.hparams.model['epochs'] is not None else self.hparams.model['steps'])
			return [optimizer], [scheduler]

class SupervisedClassifierModel(pl.LightningModule):
	def __init__(self, hparams):
		super(SupervisedClassifierModel, self).__init__()
		self.save_hyperparameters(hparams)

		self.embedding_size = self.hparams['model']['embedding_size']

		self.encoder = RecurrentEncoder(self.hparams['dataset']['type'], self.hparams['model']['num_frames'], self.embedding_size, self.hparams['model']['recurrent_block'])
		self.sequence_length = self.encoder.sequence_length
		
		if self.hparams['freeze_encoder']:
			for param in self.encoder.parameters():
					param.requires_grad = False
		
		self.linear_seperation = nn.Linear(self.encoder.embedding_size*self.encoder.sequence_length, self.hparams.dataset['num_classes'])

		self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams.dataset['num_classes'])
		self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams.dataset['num_classes'])
		self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams.dataset['num_classes'])


	def forward(self, x):
		x, _ = self.encoder(x, view_mode='flat')

		x = self.linear_seperation(x)

		return x
	
	def training_step(self, batch, batch_idx):
		x, y = batch

		y_hat = self(x)

		loss = F.cross_entropy(y_hat, y.squeeze())

		self.train_accuracy.update(y_hat, y.squeeze())

		self.log("train_loss", loss, on_step=True, prog_bar=True)

		return loss
	
	def validation_step(self, batch, batch_idx):
		if self.trainer.global_step == 0 and self.hparams.wandb: 
			wandb.define_metric('val_acc_epoch', summary='max')
			wandb.define_metric('val_loss', summary='min')
		
		x, y = batch

		y_hat = self(x)

		loss = F.cross_entropy(y_hat, y.squeeze())

		self.val_accuracy.update(y_hat, y.squeeze())

		self.log("val_loss", loss)

		return loss


	def test_step(self, batch, batch_idx):
		x, y = batch

		y_hat = self(x)

		loss = F.cross_entropy(y_hat, y.squeeze())

		self.log("test_loss", loss)

		self.test_accuracy.update(y_hat, y.squeeze())

		return loss

	def on_train_epoch_end(self):
		self.log('train_acc_epoch', self.train_accuracy.compute())
		self.train_accuracy.reset()
	
	def on_validation_epoch_end(self):
		self.log('val_acc_epoch', self.val_accuracy.compute())
		self.val_accuracy.reset()

	def on_test_epoch_end(self):
		self.log('test_acc_epoch', self.test_accuracy.compute())
		self.test_accuracy.reset()

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.model['lr'], weight_decay=self.hparams.model['weight_decay'])
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.model['epochs'] if self.hparams.model['epochs'] is not None else self.hparams.model['steps'])
		return [optimizer], [scheduler]

class LARS(optim.Optimizer):
	def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
				 weight_decay_filter=False, lars_adaptation_filter=False, logger=None):
		defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
						eta=eta, weight_decay_filter=weight_decay_filter,
						lars_adaptation_filter=lars_adaptation_filter)
		super().__init__(params, defaults)
		self.logger = logger



	def exclude_bias_and_norm(self, p):
		return p.ndim == 1

	@torch.no_grad()
	def step(self, closure=None):

		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for i, g in enumerate(self.param_groups):
			for p in g['params']:
				dp = p.grad

				if dp is None:
					continue

				if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
					dp = dp.add(p, alpha=g['weight_decay'])

				if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
					param_norm = torch.norm(p)
					update_norm = torch.norm(dp)
					one = torch.ones_like(param_norm)
					q = torch.where(param_norm > 0.,
									torch.where(update_norm > 0,
												(g['eta'] * param_norm / update_norm), one), one)
					dp = dp.mul(q)

				param_state = self.state[p]
				if 'mu' not in param_state:
					param_state['mu'] = torch.zeros_like(p)
				mu = param_state['mu']
				mu.mul_(g['momentum']).add_(dp)
				if self.logger is not None:
					self.logger(f'lr-LARS-{i}', g['lr'], on_step=True, on_epoch=False)

				p.add_(mu, alpha=-g['lr'])
		return loss
