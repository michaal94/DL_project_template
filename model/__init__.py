import os
from utils import utils
from .supervisor import Supervisor

'''
Call for the model and supervisor
Please import model and fill get_model
'''


def get_model(cfg):
    # Initialise optimiser or pass options dict
    # and initialise optimiser in Model class
    if cfg.MODEL.MODE == 'train':
        optimiser = None
    else:
        optimiser = None

    # Initialise loss
    loss = None

    if cfg.MODEL.MODEL == 'ModelName':
        # Get model here
        # I usually get model derived from nn.Module first:
        model = None
        # And after I get trainer from Model abstraction
        # trainer = Model(model, loss, optimiser)
        trainer = None
    else:
        print("No model found")
        exit()
    return trainer


def get_supervisor(cfg, model, train_loader, val_loader):
    '''
    Initialise supervisor
    '''
    logdir = cfg.LOGGING.LOGDIR
    # Add timestamp to logs directory name
    logdir = utils.timestamp_dir(logdir)
    print("Logdir path: {}".format(logdir))
    with open(os.path.join(logdir, 'config.yaml'), 'w') as f:
        print(cfg, file=f)
    # Record config path
    cfg_name = os.path.basename(cfg.INFO.CFG_PATH)
    # Make a copy of config to save all the settings
    utils.copy_file(cfg.INFO.CFG_PATH, os.path.join(logdir, cfg_name))
    # Copy all the code for 100% reproducability
    utils.mkdirs(os.path.join(logdir, 'code'))
    # Copy directories
    '''
    If you added directiories you'd like backed up add them here
    '''
    utils.copy_dirs(
        [os.path.join(cfg.INFO.PROJECT_DIR, dir_name) for dir_name in [
            'config',
            'datasets',
            'model',
            'tools',
            'utils'
        ]],
        os.path.join(logdir, 'code')
    )
    if cfg.CONFIG.CHECKPOINT_PATH != '':
        checkpoint_path = cfg.CONFIG.CHECKPOINT_PATH
    else:
        checkpoint_path = None
    sv = Supervisor(train_loader, val_loader, model, logdir=logdir,
                    epochs=cfg.CONFIG.EPOCHS,
                    tensorboard=cfg.LOGGING.TENSORBOARD,
                    debug_log=cfg.LOGGING.DEBUG,
                    debug_freq=cfg.LOGGING.DEBUG_MSG,
                    display_freq=cfg.LOGGING.DISPLAY,
                    checkpoint_freq=cfg.CONFIG.CHECKPOINT,
                    load_path=checkpoint_path,
                    zero_counters=cfg.CONFIG.ZERO_TRAINING)

    return sv
