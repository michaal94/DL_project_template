import setup

import argparse
from config import get_cfg
from datasets import get_dataloader
from model import get_model
from model import get_supervisor

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-cfg", default="default")
args = parser.parse_args()

cfg = get_cfg(args.config)

dl_train = get_dataloader(cfg, 'train')
dl_test = get_dataloader(cfg, 'test')

model = get_model(cfg)

sv = get_supervisor(cfg, model, dl_train, dl_test)

sv.train()
