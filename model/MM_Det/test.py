import torch
import os
from copy import deepcopy
from tqdm import tqdm

from options.test_options import TestOption
from utils.trainer import Trainer
from utils.utils import get_logger, get_test_dataset_configs,  set_random_seed
from dataset import get_test_dataloader, get_inf_dataloader
from builder import get_model


#image를 보고 mm_representation이 생성된 상태에서 모델 추론 모듈 따로 분리

def inference(data_path, mm_path, image_path):
    logger = get_logger(__name__, config)
    args = TestOption().parse()
    config = {'data_root': f'{data_path}',
              'ckpt': 'weights/MM-Det/current_model.pth', 
              'lmm_ckpt': 'sparklexfantasy/llava-1.5-7b-rfrd', 
              'lmm_base': None, 
              'st_ckpt': 'weights/ViT/vit_base_r50_s16_224.orig_in21k/jx_vit_base_resnet50_224_in21k-6f7c7740.pth', 
              'st_pretrained': True, 
              'model_name': 'MMDet', 'expt': 'MMDet_01', 'window_size': 10, 
              'conv_mode': 'llava_v1', 'new_tokens': 64, 'selected_layers': [-1], 
              'interval': 200, 'load_8bit': False, 'load_4bit': False, 'seed': 42, 
              'gpus': 1, 'cache_mm': True, 
              'mm_root': f'{mm_path}', 
              'debug': False, 'classes': ['inference'], 'bs': 1, 
              'mode': 'test', 'sample_size': -1, 'num_workers': 1}
    
    inf_dataset_config = {'inference': {'data_root': f'{config["data_root"]}', 
                                        'dataset_type': 'VideoFolderDatasetForReconsWithFn', 'mode': 'test', 
                                        'selected_cls_labels': [('0_real', 0)], 'sample_method': 'entire'}}
    
    set_random_seed(config['seed'])
    config['st_pretrained'] = False
    config['st_ckpt'] = None   # disable initialization
    model = get_model(config)
    model.eval()
    path = None
    if os.path.exists(config['ckpt']):
        logger.info(f'Load checkpoint from {config["ckpt"]}')
        path = config['ckpt']
    elif os.path.exists('expts', config['expt'], 'checkpoints'):
        if os.path.exists(os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')):
            logger.info(f'Load best checkpoint from {config["ckpt"]}')
            path = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')
        elif os.path.exists(os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')):
            logger.info(f'Load latest checkpoint from {config["ckpt"]}')
            path = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')
    if path is None:
        raise ValueError(f'Checkpoint not found: {config["ckpt"]}')
    state_dict = torch.load(path)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=config['cache_mm'])
    
    inf_config = deepcopy(config)
    inf_config['datasets'] = inf_dataset_config
    trainer = Trainer(
        config=inf_config, 
        model=model, 
        logger=logger,
    )
    trainer.val_dataloader = get_inf_dataloader(inf_dataset_config)
    if 'sample_size' in config:    # evaluation on sampled data to save time
        stop_count = config['sample_size']
    else:
        stop_count = -1
    results = trainer.validation_video(stop_count=stop_count)
    logger.info(f'{dataset_class}')
    for metric, value in results['metrics'].items():
        logger.info(f'{metric}: {value}')


    
# if __name__ == '__main__':
#     args = TestOption().parse()
#     config = args.__dict__

#     logger = get_logger(__name__, config)
#     logger.info(config)
#     set_random_seed(config['seed'])
#     dataset_classes = config['classes']
#     logger.info(f'Validation on {dataset_classes}.')
#     test_dataset_configs = get_test_dataset_configs(config)
#     config['st_pretrained'] = False
#     config['st_ckpt'] = None   # disable initialization
#     model = get_model(config)
#     model.eval()
#     path = None
#     if os.path.exists(config['ckpt']):
#         logger.info(f'Load checkpoint from {config["ckpt"]}')
#         path = config['ckpt']
#     elif os.path.exists('expts', config['expt'], 'checkpoints'):
#         if os.path.exists(os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')):
#             logger.info(f'Load best checkpoint from {config["ckpt"]}')
#             path = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')
#         elif os.path.exists(os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')):
#             logger.info(f'Load latest checkpoint from {config["ckpt"]}')
#             path = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')
#     if path is None:
#         raise ValueError(f'Checkpoint not found: {config["ckpt"]}')
#     state_dict = torch.load(path)
#     if 'model_state_dict' in state_dict:
#         state_dict = state_dict['model_state_dict']
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         new_state_dict[k.replace('module.', '')] = v
#     state_dict = new_state_dict
#     model.load_state_dict(state_dict, strict=config['cache_mm'])

#     for dataset_class, test_dataset_config in zip(dataset_classes, test_dataset_configs):
#         test_config = deepcopy(config)
#         test_config['datasets'] = test_dataset_config
#         trainer = Trainer(
#             config=test_config, 
#             model=model, 
#             logger=logger,
#         )
#         trainer.val_dataloader = get_test_dataloader(test_dataset_config)
#         if 'sample_size' in config:    # evaluation on sampled data to save time
#             stop_count = config['sample_size']
#         else:
#             stop_count = -1
#         results = trainer.validation_video(stop_count=stop_count)
#         logger.info(f'{dataset_class}')
#         for metric, value in results['metrics'].items():
#             logger.info(f'{metric}: {value}')
if __name__ == '__main__':
    args = TestOption().parse()
    
    config = args.__dict__

    logger = get_logger(__name__, config)
    
    inf_dataset_config = {'inference': {'data_root': f'{config["data_root"]}', 
                                        'dataset_type': 'VideoFolderDatasetForReconsWithFn', 'mode': 'test', 
                                        'selected_cls_labels': [('0_real', 0)], 'sample_method': 'entire'}}
    
    set_random_seed(config['seed'])
    config['st_pretrained'] = False
    config['st_ckpt'] = None   # disable initialization
    model = get_model(config)
    model.eval()
    path = None
    

    if os.path.exists(config['ckpt']):
        logger.info(f'Load checkpoint from {config["ckpt"]}')
        path = config['ckpt']
    elif os.path.exists('expts', config['expt'], 'checkpoints'):
        if os.path.exists(os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')):
            logger.info(f'Load best checkpoint from {config["ckpt"]}')
            path = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_best.pth')
        elif os.path.exists(os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')):
            logger.info(f'Load latest checkpoint from {config["ckpt"]}')
            path = os.path.join('expts', config['expt'], 'checkpoints', 'current_model_latest.pth')
    if path is None:
        raise ValueError(f'Checkpoint not found: {config["ckpt"]}')
    state_dict = torch.load(path)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=config['cache_mm'])
    
    inf_config = deepcopy(config)
    inf_config['datasets'] = inf_dataset_config
    breakpoint()
    trainer = Trainer(
        config=inf_config, 
        model=model, 
        logger=logger,
    )

    trainer.inference_dataloader = get_inf_dataloader(inf_dataset_config)
    if 'sample_size' in config:    # evaluation on sampled data to save time
        stop_count = config['sample_size']
    else:
        stop_count = -1
    results = trainer.inference_video()

    for metric, value in results['metrics'].items():
        logger.info(f'{metric}: {value}')