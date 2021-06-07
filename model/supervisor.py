'''
A supervisor for training
Deals with calling trainer for the model and logging stuff
'''

import os
from utils.logger import Logger


class Supervisor():
    def __init__(self, train_loader, val_loader, model,
                 epochs=100,
                 tensorboard=True,
                 logdir='../outputs/_testlog',
                 debug_log=True,
                 debug_freq=250,
                 display_freq=50,
                 checkpoint_freq=250,
                 load_path=None,
                 zero_counters=False):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model

        # Logs
        logfile = os.path.join(logdir, 'logs.log')
        self.logger = Logger(logfile)
        debug_log = os.path.join(logdir, 'debug.log')
        self.debug_log = Logger(debug_log)

        self.save_path = logdir

        self.start_iter = 0
        self.start_epoch = 0

        if load_path is not None:
            self.logger.write("Loading checkpoint from: {}".format(load_path))
            if zero_counters:
                self.logger.write("Zeroing counters and optimiser state")
            epoch, num_iter = self.model.load_checkpoint(load_path,
                                                         zero_train=zero_counters)
            if epoch is not None:
                self.start_epoch = epoch
            if num_iter is not None:
                self.start_iter = num_iter

        self.tensorboard = tensorboard
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=logdir)

        self.epochs = epochs
        self.debug_freq = debug_freq
        self.display_freq = display_freq
        self.checkpoint_freq = checkpoint_freq

        self.criterium_stat = 0.0
        # self.best_stats = None
        self.logged_stats = {
            'train_losses': [],
            'train_stats': [],
            'train_iters': [],
            'val_stats': [],
            'val_iters': []
        }

    def train(self):
        num_iter = self.start_iter
        self.logger.write("Starting training of the: {}".format(str(self.model)))
        self.log_info()
        for e in range(self.start_epoch, self.epochs):
            loss_sum = 0.0
            samples_sum = 0
            iter_cnt = 0
            for data in self.train_loader:
                iter_cnt += 1
                self.set_train()
                self.model.set_input(data)
                if num_iter % self.debug_freq == 0:
                    loss, stats = self.model.train_step(print_debug=self.debug_log)
                else:
                    loss, stats = self.model.train_step()

                if isinstance(loss, dict):
                    loss_calc = loss[list(loss.keys())[0]]
                else:
                    loss_calc = loss

                # print(loss_calc)
                if iter_cnt % len(self.train_loader) == 0:
                    multiplier = len(self.train_loader.dataset) % self.train_loader.batch_size
                    if multiplier == 0:
                        multiplier = self.train_loader.batch_size
                    batch_loss = loss_calc * multiplier
                    samples_sum = len(self.train_loader.dataset)
                else:
                    batch_loss = loss_calc * self.train_loader.batch_size
                    samples_sum += self.train_loader.batch_size
                loss_sum += batch_loss

                # print(loss_sum)
                if num_iter % self.display_freq == 0:
                    running_loss = loss_sum / samples_sum
                    msg = "Epoch: {:04d}/{:04d}\t".format(e, self.epochs)
                    msg += "Iterartion: {:06d}\t".format(num_iter)
                    if isinstance(loss, dict):
                        for k, v in loss.items():
                            msg += "{}: {:.3f}\t".format(k.capitalize(), v)
                            tag = 'Train/' + k.capitalize()
                            if self.tensorboard:
                                self.writer.add_scalar(tag, v, num_iter)
                                self.writer.flush()
                    else:
                        msg += "Loss: {:.4f}\t".format(running_loss)
                        tag = 'Train/Loss'
                        # print(running_loss, num_iter)
                        if self.tensorboard:
                            self.writer.add_scalar(tag, running_loss, num_iter)
                            self.writer.flush()
                    if stats is not None:
                        for k, v in stats.items():
                            if 'img' in k.lower():
                                print("Image")
                                if self.tensorboard:
                                    print(num_iter)
                                    self.writer.add_image(k, v, num_iter)
                            else:
                                msg += "{}: {}\t".format(k.capitalize(), v)
                    self.logger.write(msg)
                if num_iter != 0 and num_iter % self.checkpoint_freq == 0:
                    self.logger.write("==Validation==")
                    stats = self.test()
                    path = os.path.join(self.save_path, "checkpoint_{:06d}.pt".format(num_iter))
                    self.logger.write("Saving checkpoint: {}".format(path))
                    self.model.save_checkpoint(path, e, num_iter)
                    self.logger.write("Validation:")
                    if isinstance(stats, dict):
                        decision_stat = stats[list(stats.keys())[0]]
                        for k, v in stats.items():
                            self.logger.write("{}: {}".format(k, v))
                            tag = 'Val/' + k.capitalize()
                            if self.tensorboard and not isinstance(v, dict):
                                self.writer.add_scalar(tag, v, num_iter)
                                self.writer.flush()
                    else:
                        decision_stat = stats
                        self.logger.write("{}: {}".format('Acc', stats))
                        tag = 'Val/Acc'
                        if self.tensorboard:
                            self.writer.add_scalar(tag, stats, num_iter)
                            self.writer.flush()
                    if decision_stat > self.criterium_stat:
                        self.criterium_stat = decision_stat
                        self.logger.write("Best model saving")
                        path = os.path.join(self.save_path, "checkpoint_best.pt")
                        self.model.save_checkpoint(path, e, num_iter)
                num_iter += 1
        self.logger.close()
        self.debug_log.close()

    def test(self):
        self.logger.write("Testing model {}".format(str(self.model)))
        self.set_test()
        # Whole validation goes into the model - I guess it is a bit more
        # general - and frankly we don't need intermediate batch scores here
        stats = self.model.validate(self.val_loader)
        return stats

    def log_info(self):
        self.logger.write('Dataset: {}'.format(str(self.train_loader.dataset)))
        self.logger.write('Dataset length (train): {}'.format(len(self.train_loader.dataset)))
        self.logger.write('Dataloader length (train): {}'.format(len(self.train_loader)))
        self.logger.write('Dataset length (val): {}'.format(len(self.val_loader.dataset)))
        self.logger.write('Dataloader length (val): {}'.format(len(self.val_loader)))

    def set_train(self):
        self.model.set_train()

    def set_test(self):
        self.model.set_test()
