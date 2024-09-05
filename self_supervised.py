import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger
from dataset import data_loader
from models import SSLModel
import argparse

def get_config():
	parser = argparse.ArgumentParser(description='Self-supervised learning script')
	parser.add_argument('--debug', action='store_true', help='Enable debug mode')
	parser.add_argument('--slurm', action='store_true', help='Enable slurm mode')
	parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--num_workers', type=int, default=24, help='Number of workers for data loading')
	parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
	parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
	parser.add_argument('--embedding_size', type=int, default=512, help='Embedding size for encoder')
	parser.add_argument('--num_frames', type=int, default=50, help='Number of frames for each sample')
	parser.add_argument('--projection_size', type=int, default=256, help='Projection size for encoder')
	parser.add_argument('--projection_size_last_layer', type=int, default=None, help='Last layer of projection size for encoder')
	parser.add_argument('--shared-weights', action='store_true', help='Shared weights')
	parser.add_argument('--database', type=str, choices=['SignFi', 'UT_HAR'], default='SignFi', help='Type of database')
	parser.add_argument('--loss', type=str, choices=['BT', 'AutoFi', 'VICReg', 'SimCLR', 'CPC', 'CPC+BT', 'CPC+SimCLR', 'CPC+AutoFi', 'CPC+VICReg'], default='BT', help='Type of loss')
	parser.add_argument('--lambd', type=float, default=0.002, help='Lambda for BT loss')
	parser.add_argument('--timestep', type=int, default=3, help='Timestep for CPC')
	parser.add_argument('--projector-less', action='store_true', help='Remove projector')
	parser.add_argument('--cpc-coeff', type=float, default=None, help='CPC coefficient')
	parser.add_argument('--cpc-autoregressive-model', type=str, default='GRU', choices=['GRU', 'LSTM', 'RNN'], help='CPC autoregressive model')
	
	# Augmentations
	parser.add_argument('--aug-noise', action='store_true', help='Enable noise augmentation')
	parser.add_argument('--aug-time-flip', action='store_true', help='Enable time flip augmentation')
	parser.add_argument('--aug-time-masking', action='store_true', help='Enable time masking augmentation')
	parser.add_argument('--aug-subcarrier-masking', action='store_true', help='Enable subcarrier masking augmentation')
	parser.add_argument('--aug-random-dual', action='store_true', help='Enable SignFi random dual augmentation')
	parser.add_argument('--aug-temporal-predictive', action='store_true', help='Enable temporal predictive augmentation')


	args = parser.parse_args()

	loss_map = {
		'BT': ['barlow_twin'],
		'AutoFi': ['probability_consistency', 'mutual_information', 'geometric_consistency'],
		'VICReg': ['invariance', 'variance', 'covariance'],
		'SimCLR': ['SimCLR'],
		'CPC': ['CPC'],
		'CPC+BT': ['CPC', 'barlow_twin'],
		'CPC+SimCLR': ['CPC', 'SimCLR'],
		'CPC+AutoFi': ['CPC', 'probability_consistency', 'mutual_information', 'geometric_consistency'],
		'CPC+VICReg': ['CPC', 'invariance', 'variance', 'covariance'],
	}

	augmentations = {}

	if args.aug_noise:
		augmentations['noise'] = [0, 0.1]
	if args.aug_time_flip:
		augmentations['flip'] = ['time']
	if args.aug_time_masking:
		augmentations['zero_masking'] = {
			'dim': ['time'],
			'ratio': [0.10],
		}
	if args.aug_subcarrier_masking:
		if 'zero_masking' in augmentations:
			augmentations['zero_masking']['dim'].append('subcarrier')
			augmentations['zero_masking']['ratio'].append(0.10)
		else:
			augmentations['zero_masking'] = {
				'dim': ['subcarrier'],
				'ratio': [0.10],
			}
	if args.aug_random_dual:
		augmentations['SignFiRandomDual'] = True
	if args.aug_temporal_predictive:
		augmentations['temporal_predictive'] = True
	
	if len(augmentations) == 0:
		augmentations = {
			'noise': [0, 0.1],
			'flip': ['time'],
			'zero_masking': {
				'dim': ['subcarrier'],
				'ratio': [0.10],
			},
			'SignFiRandomDual': True,
			'temporal_predictive': True
		}

		if args.loss == 'AutoFi':
			augmentations['SignFiRandomDual'] = False
			augmentations['temporal_predictive'] = False
			augmentations['zero_masking'] = False
		elif args.loss == 'VICReg':
			augmentations['zero_masking'] = False
		elif args.loss == 'SimCLR':
			augmentations = {
				'zero_masking': {
					'dim': ['time'],
					'ratio': [0.10],
				}
			}
		elif args.loss == 'BT':
			augmentations['zero_masking'] = {
				'dim': ['time'],
				'ratio': [0.10],
			}
			augmentations['SignFiRandomDual'] = False
			augmentations['flip'] = False
			augmentations['temporal_predictive'] = False
		elif args.loss == 'CPC':
			augmentations['zero_masking'] = False
			augmentations['SignFiRandomDual'] = False
			augmentations['flip'] = False
			augmentations['noise'] = False
			augmentations['temporal_predictive'] = False
		elif args.loss == 'CPC+BT' or args.loss == 'CPC+SimCLR' or args.loss == 'CPC+AutoFi':
			augmentations['zero_masking'] = False
			augmentations['temporal_predictive'] = False
			augmentations['flip'] = False
		elif args.loss == 'CPC+VICReg':
			augmentations['zero_masking'] = False
			augmentations['temporal_predictive'] = False
		else:
			raise NotImplementedError
	if args.database == 'SignFi':
		dataset = {
			'root_dir': '/local/data0/Borna/Projects/SignFi Dataset/',
			'batch_size': args.batch_size,
			'type': 'SignFi',
			'name': 'SignFi_Lab',
			'local': False,
			'num_classes': 276,
			'SignFi_env': 'lab',
			'SignFi_link': 'all',
			'SignFi_mode': 'dual' if 'SignFiRandomDual' in augmentations and augmentations['SignFiRandomDual'] else 'single',
			'mode': 'all_data',
			'input_shape': (1, 3, 30, 200),
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
			}
		}

	
	cfg = {
		'dataset': dataset,
		'model': {
			'lr': 0.2,
			# 'lr': 1e-2,
			'lr_biases': 0.0048,
			# 'weight_decay': 5e-4,
			'weight_decay': 1.5e-6,
			'momentum': 0.9,
			'lambd': args.lambd if args.lambd is not None else 0.0051,
			'timestep': args.timestep,
			'cpc_coeff': args.cpc_coeff if args.cpc_coeff is not None else 50 if args.loss == "CPC+BT" or args.loss == "CPC+VICReg" else 100 if args.loss == "CPC+SimCLR" else 0.01 if args.loss == "CPC+AutoFi" else 1,
			'cpc_autoregressive_model': args.cpc_autoregressive_model,
			'epochs': args.epochs,
			'sim_coeff': 25,
			'std_coeff': 25,
			'cov_coeff': 2,
			'temperature': 0.5,
			'EPS': 1e-4,
			'n_hidden_states_nodes': args.projection_size,
			'n_hidden_states_nodes_last_layer': args.projection_size_last_layer if args.projection_size_last_layer is not None else args.projection_size,
			'remove_projector': args.projector_less,
			'embedding_size': args.embedding_size,
			'losses': loss_map[args.loss],
			'shared_weights': args.shared_weights,
			'augmentations': augmentations,
			'num_frames': args.num_frames,
			'recurrent_block': False
		},
		'seed': args.seed,
		'num_workers': args.num_workers,
		'debug': args.debug,
		'slurm': args.slurm,
	}

	if cfg['slurm']:
		slurm_tmpdir = os.getenv('SLURM_TMPDIR')
		if args.database == 'SignFi':
			cfg['dataset']['root_dir'] = os.path.join(slurm_tmpdir, 'data/SignFi Dataset/')
		elif args.database == 'UT_HAR':
			cfg['dataset']['root_dir'] = os.path.join(slurm_tmpdir, 'data/')
		else:
			raise NotImplementedError
	
	if not cfg['debug']:	
		if args.wandb_name is not None:
			wandb.init(config=cfg, project='SSL-Sensing', name=args.wandb_name)
		else:
			wandb.init(config=cfg, project='SSL-Sensing')
		# Config parameters are automatically set by W&B sweep agent
		cfg = wandb.config.as_dict()

	return cfg

def main(cfg):
	seed_everything(cfg['seed'], workers=True)

	train_loader, validation_loader, test_loader, unsupervised_loader = data_loader(cfg['dataset'], cfg['num_workers'])


	if not cfg['debug']:
		wandb_logger = WandbLogger(project='SSL-Sensing')
		checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode="min")

		wandb_logger.experiment.config.update(cfg)

	

	model = SSLModel(cfg)

	if cfg['dataset']['type'] == 'SignFi':
		model = model.double()
	
	if cfg['debug']:
		trainer = Trainer(
			devices="auto",
			accelerator="auto",
			fast_dev_run=2,
			# overfit_batches=1,
			detect_anomaly=True,
			gradient_clip_val=0.8,
			max_epochs=cfg['model']['epochs'],
			log_every_n_steps=1,
			# val_check_interval=0,
		)
	else:
		trainer = Trainer(
			devices="auto",
			accelerator="auto",
			# overfit_batches=1,
			detect_anomaly=True,
			gradient_clip_val=0.8,
			max_epochs=cfg['model']['epochs'],
			log_every_n_steps=1,
			# val_check_interval=0,
			logger=wandb_logger,
			callbacks=[checkpoint_callback],
		)
	
	trainer.fit(model, train_dataloaders=unsupervised_loader)

	if not cfg['debug']:
		# Print the best model's path
		print(f"Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
	cfg = get_config()
	main(cfg)
	