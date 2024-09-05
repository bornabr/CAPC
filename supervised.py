import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
from dataset import data_loader
from models import SSLModel, LinearClassifierModel

import argparse

def get_config():

	parser = argparse.ArgumentParser(description='Self-supervised learning script')
	parser.add_argument('--debug', action='store_true', help='Enable debug mode')
	parser.add_argument('--slurm', action='store_true', help='Enable slurm mode')
	parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--num_workers', type=int, default=24, help='Number of workers for data loading')
	parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
	parser.add_argument('--ssl_model', type=str, default=None, help='SSL model checkpoint path')
	parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
	parser.add_argument('--steps', type=int, default=None, help='Number of steps for training')
	parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights')
	parser.add_argument('--partial_freeze_encoder', action='store_true', help='Freeze encoder weights partially')
	parser.add_argument('--supervised', action='store_true', help='Enable supervised mode')
	parser.add_argument('--embedding_size', type=int, default=None, help='Embedding size for encoder output in supervised mode')
	parser.add_argument('--num_frames', type=int, default=None, help='Number of frames to use for each sample')
	parser.add_argument('--recurrent_block', type=int, default=None, help='Enable recurrent block in encoder and its size')
	parser.add_argument('--portion', type=int, default=None, help='Portion of samples per class to use')
	parser.add_argument('--semi-supervised', action='store_true', help='Enable semi-supervised mode')
	parser.add_argument('--protonet', action='store_true', help='Enable protonet mode')
	parser.add_argument('--database', type=str, choices=['SignFi', 'UT_HAR'], default='SignFi', help='Type of database')
	parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('--lr-encoder', type=float, default=None, help='Learning rate Encoder')


	args = parser.parse_args()

	if args.database == 'SignFi':
		dataset = {
			'root_dir': '/local/data0/Borna/Projects/SignFi Dataset/',
			'batch_size': args.batch_size,
			'type': 'SignFi',
			'name': 'SignFi_Home',
			'local': False,
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
			'root_dir': '/home/bornab/Projects/WiFi-CSI-Sensing-Benchmark/Data/',
			'batch_size': args.batch_size,
			'type': 'UT_HAR',
			'name': 'UT_HAR',
			'num_classes': 7,
			'local': False,
			'input_shape': (-1, 1, 250, 90),
			'dimension_maps': {
				'anttena': '3',
				'subcarrier': '3',
				'time': '2',
			},
			'portion': args.portion,
		}

	cfg = {
		'dataset': dataset,
		'ssl_model': args.ssl_model,
		'model': {
			'lr': args.lr,
			'lr_encoder': args.lr_encoder,
			'weight_decay': 5e-4,
			'epochs': args.epochs if args.steps is None else None,
			'steps': args.steps,
		},
		'seed': args.seed,
		'num_workers': args.num_workers,
		'debug': args.debug,
		'slurm': args.slurm,
		'supervised': args.supervised,
		'freeze_encoder': args.freeze_encoder if args.supervised else True,
		'partial_freeze_encoder': args.partial_freeze_encoder,
		'semi_supervised': args.semi_supervised,
	}

	if not cfg['supervised'] and cfg['ssl_model'] is None:
		raise Exception('SSL model checkpoint path is required')

	if cfg['supervised']:
		cfg['model']['embedding_size'] = args.embedding_size
		cfg['model']['recurrent_block'] = args.recurrent_block
		cfg['model']['num_frames'] = args.num_frames


	if cfg['slurm']:
		slurm_tmpdir = os.getenv('SLURM_TMPDIR')
		if args.database == 'SignFi' or args.database == 'SignFiLabSame':
			cfg['dataset']['root_dir'] = os.path.join(slurm_tmpdir, 'data/SignFi Dataset/')
		elif args.database == 'UT_HAR':
			cfg['dataset']['root_dir'] = os.path.join(slurm_tmpdir, 'data/')
		else:
			raise NotImplementedError
	

	if not cfg['debug']:
		if args.wandb_name is not None:
			wandb.init(config=cfg, project='SSL-Sensing-FineTune', name=args.wandb_name)
		else:
			wandb.init(config=cfg, project='SSL-Sensing-FineTune')
		# Config parameters are automatically set by W&B sweep agent
		cfg = wandb.config.as_dict()

	return cfg

# def print_summary(model, cfg):
# 	input_shape = tuple(cfg['dataset']['input_shape'][1:])

# 	model.to('cuda')
# 	summary(model, input_shape, device='cuda')

def main(cfg):
	seed_everything(cfg['seed'], workers=True)

	train_loader, validation_loader, test_loader, _ = data_loader(cfg['dataset'], cfg['num_workers'])


	if not cfg['debug']:
		wandb_logger = WandbLogger(project='SSL-Sensing-FineTune')
		checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

		wandb_logger.experiment.config.update(cfg)

	if not cfg['supervised']:
		# Get SSL model from checkpoint
		ssl_model = SSLModel.load_from_checkpoint(cfg['ssl_model'])
		# Get model
		model = LinearClassifierModel(ssl_model.encoder, cfg)

	# print_summary(model, cfg)

	if cfg['dataset']['type'] == 'SignFi':
		model = model.double()
	

	# Leanring rate Monitor
	lr_logger = LearningRateMonitor(logging_interval='step')

	if cfg['debug']:
		trainer = Trainer(
			devices="auto",
			accelerator="auto",
			fast_dev_run=2,
			# overfit_batches=1,
			detect_anomaly=True,
			max_epochs=cfg['model']['epochs'],
			log_every_n_steps=1,
			# val_check_interval=0,
			callbacks=[lr_logger],
		)
	else:
		early_stop_callback = EarlyStopping(
			monitor='val_loss',
			min_delta=0.00,
			patience=50,
			mode='min'
		)
		if cfg['model']['steps']:
			trainer = Trainer(
				devices="auto",
				accelerator="auto",
				# fast_dev_run=2,
				# overfit_batches=1,
				detect_anomaly=True,
				max_steps=cfg['model']['steps'],
				log_every_n_steps=1,
				# val_check_interval=0,
				logger=wandb_logger,
				callbacks=[lr_logger, checkpoint_callback, early_stop_callback],
				num_sanity_val_steps=0 if cfg['protonet'] else 2
			)
		else:
			trainer = Trainer(
				devices="auto",
				accelerator="auto",
				# fast_dev_run=2,
				# overfit_batches=1,
				detect_anomaly=True,
				max_epochs=cfg['model']['epochs'],
				log_every_n_steps=1,
				# val_check_interval=0,
				logger=wandb_logger,
				callbacks=[lr_logger, checkpoint_callback],
				num_sanity_val_steps=0 if cfg['protonet'] else 2
			)
	
	trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

	if not cfg['debug']:
		trainer.test(ckpt_path="best",dataloaders=test_loader)

if __name__ == '__main__':
	cfg = get_config()
	main(cfg)
	