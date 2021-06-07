from torch.utils.data import DataLoader

'''
Call for dataloder
Please import dataset and fill get_dataset
Note that you can use config options
'''


def get_dataset(cfg, split):
    if cfg.DATASET.DATASET == 'DatasetName':
        dataset = None
    else:
        raise NotImplementedError('No other datasets implemented yet')

    return dataset


def get_dataloader(cfg, split):
    # Feel free to adjust to your needs
    dataset = get_dataset(cfg, split)
    shuffle = cfg.MODEL.SHUFFLE if split == 'train' else False
    batch = cfg.CONFIG.BATCH if split == 'train' else cfg.CONFIG.BATCH_VAL
    loader = DataLoader(dataset=dataset, batch_size=batch,
                        shuffle=shuffle, num_workers=cfg.CONFIG.NUM_WORKERS)
    print("Loaded {} dataset, split: {} number of samples: {}".format(
        cfg.DATASET.DATASET,
        split,
        len(dataset)
    ))

    return loader
