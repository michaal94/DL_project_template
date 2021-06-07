class Model(object):
    '''
    An abstract class representing a model, provides abstractions
    for required methods
    Note the init construction:
    '''

    # def __init__(self, model, loss, optimiser):
    #     self.model = model

    #     self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #     self.model = self.model.to(self.device)
    #     self.criterion = loss

    #     if optimiser is not None:
    #         self.optimiser = (...)
    #     else:
    #         self.optimiser = None

    #     self.data = None

    def train_step(self, batch, print_debug=None):
        '''
        Perform a step of training
        As a guide:

        assert self.data is not None, "Set input first"
        self.set_train()
        self.optimiser.zero_grad()
        _ = self.model()
        self.loss = self.criterion()
        self.loss.backward()
        self.optimiser.step()
        loss_value = self.loss.data.cpu().numpy().astype(float)

        stats = None
        if print_debug:
            stats = {}
            # Fill with debug stats
        return loss_value, stats
        '''
        raise NotImplementedError

    def validate(self, loader):
        '''
        Ensure it returns dict of stats
        Ensure first entry is decisive (decides that one model is better than another)
        '''
        raise NotImplementedError

    def save_checkpoint(self):
        # Example - input: (self, path, epoch=None, num_iter=None)

        # checkpoint = {
        #     'state_dict': self.model.cpu().state_dict(),
        #     'epoch': epoch,
        #     'num_iter': num_iter
        # }
        # if self.model.training:
        #     checkpoint['optim_state_dict'] = self.optimiser.state_dict()
        # torch.save(checkpoint, path)
        # self.model = self.model.to(self.device)

        raise NotImplementedError

    def load_checkpoint(self):
        # Example - input: (self, path, zero_train=False)

        # checkpoint = torch.load(path)
        # self.model.load_state_dict(checkpoint['state_dict'])
        # if self.optimiser is not None:
        #     if self.model.training and not zero_train:
        #         self.optimiser.load_state_dict(checkpoint['optim_state_dict'])
        # epoch = checkpoint['epoch'] if not zero_train else None
        # num_iter = checkpoint['num_iter'] if not zero_train else None
        # return epoch, num_iter

        raise NotImplementedError

    def set_train(self):
        if not self.model.training:
            self.model.train()

    def set_test(self):
        if self.model.training:
            self.model.eval()

    def set_input(self, data):
        self.data = data

    def unset_input(self):
        self.data = None

    def __str__(self):
        return "Generic Model Trainer abstraction"
