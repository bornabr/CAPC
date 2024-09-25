import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
from dataset import data_loader
from models import SSLModel, LinearClassifierModel, SupervisedClassifierModel

import argparse

def get_args():
	"""Returns the arguments for the script"""

	parser = argparse.ArgumentParser(description='Self-supervised learning script')
	parser.add_argument('--database-path', type=str, default=None, help='Path to the database')
	parser.add_argument('--debug', action='store_true', help='Enable debug mode')
	parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
	parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--num_workers', type=int, default=24, help='Number of workers for data loading')
	parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
	parser.add_argument('--ssl_model', type=str, default=None, help='SSL model checkpoint path')
	parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
	parser.add_argument('--supervised', action='store_true', help='Enable supervised mode')
	parser.add_argument('--embedding_size', type=int, default=None, help='Embedding size for encoder output in supervised mode')
	parser.add_argument('--num_frames', type=int, default=None, help='Number of frames to use for each sample in supervised mode')
	parser.add_argument('--recurrent_block', type=int, default=None, help='Enable recurrent block in encoder and its size')
	parser.add_argument('--portion', type=int, default=None, help='Portion of samples per class to use (shots)')
	parser.add_argument('--semi-supervised', action='store_true', help='Enable semi-supervised mode')
	parser.add_argument('--database', type=str, choices=['SignFi', 'UT_HAR'], default='SignFi', help='Type of database')
	parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
	parser.add_argument('--lr-encoder', type=float, default=None, help='Learning rate Encoder for semi-supervised mode')

	args = parser.parse_args()

	return args

def get_database_config(args):

	if args.database == 'SignFi':
		dataset = {
			'root_dir': '/local/data0/Borna/Projects/SignFi Dataset/' if args.database_path is None else args.database_path,
			'batch_size': args.batch_size,
			'type': 'SignFi',
			'name': 'SignFi_Home',
			'num_classes': 276,
			'SignFi_env': 'home',
			'SignFi_link': 'all',
			'SignFi_mode': 'single',
			'mode': 'train_data',
			'portion': args.portion,
			'input_shape': (-1, 3, 30, 200),
			'dimension_maps': {
				'anttena': '1',
				'subcarrier': '2',
				'time': '3',
			}
		}
	elif args.database == 'UT_HAR':
		dataset = {
			'root_dir': '/home/bornab/Projects/WiFi-CSI-Sensing-Benchmark/Data/' if args.database_path is None else args.database_path,
			'batch_size': args.batch_size,
			'type': 'UT_HAR',
			'name': 'UT_HAR',
			'num_classes': 7,
			'input_shape': (-1, 1, 250, 90),
			'dimension_maps': {
				'anttena': '3',
				'subcarrier': '3',
				'time': '2',
			},
			'portion': args.portion,
		}
	else:
		raise NotImplementedError
	
	return dataset


def get_config():

	args = get_args()

	dataset = get_database_config(args)

	cfg = {
		'dataset': dataset,
		'ssl_model': args.ssl_model,
		'model': {
			'lr': args.lr,
			'lr_encoder': None if not args.semi_supervised else args.lr_encoder,
			'weight_decay': 5e-4,
			'epochs': args.epochs
		},
		'seed': args.seed,
		'num_workers': args.num_workers,
		'debug': args.debug,
		'wandb': args.wandb and not args.debug,
		'supervised': args.supervised,
		'freeze_encoder': False if args.semi_supervised or args.supervised else True,
		'semi_supervised': args.semi_supervised,
	}

	# Linear Evaluation:
	# 	Freeze encoder: True
	# 	Semi-supervised: False
	# Semi-supervised Evaluation:
	# 	Freeze encoder: False
	# 	Semi-supervised: True
	#   lr_encoder is required
	# Supervised Baseline:
	# 	Freeze encoder: False
	# 	Semi-supervised: False
	# 	embedding_size is required
	# 	num_frames is required
	# 	recurrent_block is required


	if not cfg['supervised'] and cfg['ssl_model'] is None:
		raise Exception('SSL model checkpoint path is required')

	if cfg['semi_supervised'] and cfg['model']['lr_encoder'] is None:
		raise Exception('Encoder learning rate is required for semi-supervised mode (hint: default is 5e-3)')

	if cfg['supervised']:
		cfg['model']['embedding_size'] = args.embedding_size
		cfg['model']['recurrent_block'] = args.recurrent_block
		cfg['model']['num_frames'] = args.num_frames

		if cfg['model']['embedding_size'] is None:
			raise Exception('Embedding size is required for supervised mode (hint: default is 128)')
		
		if cfg['model']['num_frames'] is None:
			raise Exception('Number of frames is required for supervised mode (hint: default is 10)')

	if cfg['wandb']:
		if args.wandb_name is not None:
			wandb.init(config=cfg, project='CAPC-Evaluation', name=args.wandb_name)
		else:
			wandb.init(config=cfg, project='CAPC-Evaluation')
		
		# Config parameters are automatically set by W&B sweep agent
		cfg = wandb.config.as_dict()

	return cfg

def main(cfg):
	seed_everything(cfg['seed'], workers=True)

	train_loader, validation_loader, test_loader, _ = data_loader(cfg['dataset'], cfg['num_workers'])


	if cfg['wandb']:
		wandb_logger = WandbLogger(project='CAPC-Evaluation')

		wandb_logger.experiment.config.update(cfg)
	
	if cfg['debug']:
		checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

	if not cfg['supervised']:
		# Get SSL model from checkpoint
		ssl_model = SSLModel.load_from_checkpoint(cfg['ssl_model'])
		# Get model
		model = LinearClassifierModel(ssl_model.encoder, cfg)
	else:
		# Get random initialized encoder and linear classifier model for supervised baseline
		model = SupervisedClassifierModel(cfg)

	if cfg['dataset']['type'] == 'SignFi':
		model = model.double()	

	# Leanring rate Monitor
	lr_logger = LearningRateMonitor(logging_interval='step')

	if cfg['debug']:
		trainer = Trainer(
			devices="auto",
			accelerator="auto",
			fast_dev_run=2,
			max_epochs=cfg['model']['epochs'],
			log_every_n_steps=1,
			callbacks=[lr_logger],
		)
	else:
		early_stop_callback = EarlyStopping(
			monitor='val_loss',
			min_delta=0.00,
			patience=50,
			mode='min'
		)
		if cfg['wandb']:
			trainer = Trainer(
				devices="auto",
				accelerator="auto",
				max_epochs=cfg['model']['epochs'],
				log_every_n_steps=1,
				logger=wandb_logger,
				callbacks=[lr_logger, checkpoint_callback],
				num_sanity_val_steps= 2
			)
		else:
			trainer = Trainer(
				devices="auto",
				accelerator="auto",
				max_epochs=cfg['model']['epochs'],
				log_every_n_steps=1,
				callbacks=[lr_logger, early_stop_callback],
				num_sanity_val_steps= 2
			)

	trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

	if not cfg['debug']:
		trainer.test(ckpt_path="best",dataloaders=test_loader)

if __name__ == '__main__':
	cfg = get_config()
	main(cfg)