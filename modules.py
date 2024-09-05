import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
	def __init__(self, width):
		super(EncoderBlock, self).__init__()
		self.width = width
		
		self.conv1 = nn.Sequential(
			nn.Conv2d(self.width, self.width, kernel_size=(3, 1), padding='same',dilation=(1, 1)),  # 3x1 2D Convolution (d=1)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),
			nn.Conv2d(self.width, self.width, kernel_size=(1, 3), padding='same',dilation=(1, 1)),  # 1x3 2D Convolution (d=1)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),

			nn.Conv2d(self.width, self.width, kernel_size=(3, 1), padding='same',dilation=(2, 2)),  # 3x1 2D Convolution (d=2)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),
			nn.Conv2d(self.width, self.width, kernel_size=(1, 3), padding='same',dilation=(2, 2)),  # 1x3 2D Convolution (d=2)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),

			nn.Conv2d(self.width, self.width, kernel_size=(3, 1), padding='same',dilation=(3, 3)),  # 3x1 2D Convolution (d=3)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),
			nn.Conv2d(self.width, self.width, kernel_size=(1, 3), padding='same',dilation=(3, 3)),  # 1x3 2D Convolution (d=3)
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),			
		)


		self.conv2 = nn.Sequential(
			nn.Conv2d(self.width,self.width, kernel_size=(3, 3), padding='same'),
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3)
		)

		self.prelu1 = nn.PReLU(num_parameters=2*self.width, init=0.3)

		self.conv1x1 = nn.Sequential(
			nn.Conv2d(2*self.width,self.width, kernel_size=(1, 1), padding='same'),
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3)
		)

		self.prelu2 = nn.PReLU(num_parameters=self.width, init=0.3)

		self.Identity = nn.Identity()
	
	def forward(self, x):
		identity = self.Identity(x)
		res1=self.conv1(x)
		res2=self.conv2(x)
		res=self.prelu1(torch.cat((res1,res2),dim=1))
		res=self.conv1x1(res)
		return self.prelu2(identity + res)

class RecurrentBlock(nn.Module):
	def __init__(self, input_size, hidden_size, keep_dim=False):
		super(RecurrentBlock, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.recurrent_block = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
		self.keep_dim = keep_dim
		if keep_dim:
			self.fc = nn.Linear(self.hidden_size, self.input_size)
	
	def forward(self, x):
		x, hidden = self.recurrent_block(x)
		if self.keep_dim:
			x = self.fc(x)
		return x, hidden

class Encoder(nn.Module):
	def __init__(self, input_shape, embedding_size):
		super(Encoder, self).__init__()
		self.input_size = np.prod(input_shape)
		self.width = input_shape[1]
		self.embedding_size = embedding_size

		self.encoder = nn.Sequential(
			nn.Conv2d(self.width,self.width, kernel_size=(5,5), padding='same'),
			nn.BatchNorm2d(self.width),
			nn.PReLU(num_parameters=self.width, init=0.3),
			EncoderBlock(self.width),
		)
		self.encoder_fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.input_size, embedding_size),
		)
	
	def forward(self, x):
		x = self.encoder(x)
		x = self.encoder_fc(x)
		return x

class RecurrentEncoder(Encoder):
	def __init__(self, input_type, num_frames, embedding_size, recurrent_block=False, augmentations=None):
		self.input_type = input_type
		self.num_frames = num_frames
		self.augmentations = augmentations
		if input_type == 'UT_HAR':
			self.input_shape = (1, 1, num_frames, 90)
			self.sequence_length = 250//num_frames
		elif input_type == 'SignFi':
			self.input_shape = (1, 3, num_frames, 30)
			self.sequence_length = 200//num_frames
		else:
			raise NotImplementedError
		


		super(RecurrentEncoder, self).__init__(self.input_shape, embedding_size)

		self.recurrent_block = recurrent_block
		if self.recurrent_block:
			self.lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True)
	
	def zero_masking(self, tensor):
		ratios = self.augmentations['zero_masking']['ratio']
		dims = self.augmentations['zero_masking']['dim']

		if not isinstance(dims, list):
			dims = [dims]
			ratios = [ratios]

		for dim, ratio in zip(dims, ratios):
			dim_int = None

			if type(dim) is not int:
				if dim == 'time':
					dim_int = 2
				elif dim == 'antenna':
					dim_int = 1
				elif dim == 'subcarrier':
					dim_int = 3
				else:
					raise NotImplementedError
			else:
				dim_int = dim

			batch_size = tensor.shape[0]

			mask_length = int(ratio * tensor.shape[dim_int])

			if dim == 'time' or dim == 'subcarrier':	
				# Randomly choose the start index for zero masking for each sample in the batch
				start_idx = torch.randint(0, tensor.size(dim_int) - mask_length + 1, (batch_size,))
				# Create a mask of ones
				mask = torch.ones_like(tensor)

				# Create a tensor of indices for the batch dimension
				batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, mask_length)

				# Create a tensor of indices for the mask dimension
				mask_indices = start_idx.unsqueeze(1) + torch.arange(mask_length)

				# Set the segment to zero in the mask using advanced indexing
				if dim_int == 1:
					mask[batch_indices, mask_indices, ...] = 0
				elif dim_int == 2:
					mask[batch_indices, :, mask_indices, ...] = 0
				elif dim_int == 3:
					mask[batch_indices, :, :, mask_indices, ...] = 0
				# Extend for more dimensions if necessary...

				# Apply mask
				tensor = tensor * mask
			elif dim == 'antenna':
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
			else:
				raise NotImplementedError

		return tensor	

	def augmentation(self, x):
		if 'zero_masking' in self.augmentations and self.augmentations['zero_masking']:
			x = self.zero_masking(x)
		return x

	def forward(self, x, view_mode='in_sequence', output='all', mode='supervised'):
		batch_size = x.shape[0]
		if self.input_type == 'UT_HAR':
			# batch x 1 x 250 x 90
			new_x = x.permute(0,2,1,3).contiguous()
			# batch x 250 x 1 x 90
			new_x = new_x.view(batch_size*self.sequence_length,self.num_frames,1,90)
			# (batch x t) x num_frames x 1 x 90
			new_x=new_x.permute(0,2,1,3)
			#(batch x t)x 1 x num_frames x 90
		elif self.input_type == 'SignFi':
			# batch x 3 x 30 x 200
			new_x = x.permute(0,3,1,2).contiguous()
			# batch x 200 x 3 x 30
			new_x = new_x.view(batch_size*self.sequence_length,self.num_frames,3,30)
			# (batch x t) x num_frames x 3 x 30
			new_x=new_x.permute(0,2,1,3)
			#(batch x t)x 3 x num_frames x 30
		else:
			raise NotImplementedError
		
		if self.augmentations is not None and mode == 'self_supervised':
			new_x = self.augmentation(new_x)

		new_x = super(RecurrentEncoder, self).forward(new_x)
		# (batch x t) x embedding_size

		if self.recurrent_block:
			new_x = new_x.view(batch_size, self.sequence_length, self.embedding_size)
			# batch x t x embedding_size

			new_x, _ = self.lstm(new_x)

			if output == 'last':
				new_x = new_x[:, -1, :]
				# batch x embedding_size

				return new_x, self.embedding_size
			elif output == 'all':
				if view_mode == 'in_sequence':
					return new_x, self.sequence_length
				elif view_mode == 'in_batch':
					new_x = new_x.contiguous()
					return new_x.view(batch_size*self.sequence_length, self.embedding_size), batch_size*self.sequence_length
				elif view_mode == 'flat':
					new_x = new_x.contiguous()
					return new_x.view(batch_size, self.sequence_length*self.embedding_size), self.embedding_size*self.sequence_length
				else:
					raise NotImplementedError
			else:
				raise NotImplementedError
		else:
			if view_mode == 'in_sequence':
				return new_x.view(batch_size, self.sequence_length, self.embedding_size), self.sequence_length
			elif view_mode == 'in_batch':
				return new_x, batch_size*self.sequence_length
			elif view_mode == 'flat':
				return new_x.view(batch_size, self.sequence_length*self.embedding_size), self.embedding_size*self.sequence_length
			else:
				raise NotImplementedError
		

class projector(nn.Module):
	def __init__(self, hidden_states=256, hidden_states_last_layer=256, embedding_size=256):
		super(projector, self).__init__()
		
		self.fc1 = nn.Sequential(
			nn.Linear(embedding_size, hidden_states),
			nn.BatchNorm1d(hidden_states),
			nn.ReLU(inplace=True)
		)

		self.fc2 = nn.Sequential(
			nn.Linear(hidden_states, hidden_states),
			nn.BatchNorm1d(hidden_states),
			nn.ReLU(inplace=True)
		)

		self.fc3 = nn.Linear(hidden_states, hidden_states_last_layer, bias=False)

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return x
