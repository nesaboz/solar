import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize
import random
from tqdm import tqdm
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from torchviz import make_dot
from torch.utils.data import TensorDataset


RUNS_FOLDER_NAME = 'runs'


def make_lr_fn(start_lr, end_lr, num_iter, step_mode='exp'):
    if step_mode == 'linear':
        factor = (end_lr / start_lr - 1) / num_iter

        def lr_fn(iteration):
            return 1 + iteration * factor
    else:
        factor = (np.log(end_lr) - np.log(start_lr)) / num_iter

        def lr_fn(iteration):
            return np.exp(factor) ** iteration
    return lr_fn


class StepByStep(object):
    """
    Main class for training neural network.

    Args:
        model:
        optimizer:
        loss_fn:
        device:

    """

    def __init__(self, model, optimizer, loss_fn):
        super(StepByStep, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.scheduler = None
        self.is_batch_lr_scheduler = False

        self.total_epochs = 0

        self.losses = []
        self.val_losses = []
        self.learning_rates = []

        self.visualization = {}
        self.handles = {}

    def __str__(self):
        return f"{self.model}, {self.optimizer}, {self.loss_fn}"

    __repr__ = __str__

    def set_loaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def perform_train_step(self, batch_x, batch_y):
        self.model.train()

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        predictions = self.model(batch_x)
        loss = self.loss_fn(predictions, batch_y)  # order is critical ofcourse, first are predictions then true labels.
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def perform_val_step(self, batch_x, batch_y):
        self.model.eval()

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        predictions = self.model(batch_x)
        loss = self.loss_fn(predictions, batch_y)

        return loss.item()

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        self.device = device
        self.model.to(self.device)

    def set_tensorboard(self, name, folder=RUNS_FOLDER_NAME):
        # This method allows the user to define a SummaryWriter to interface with TensorBoard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def check_consistency(self):
        assert len(list(self.model.parameters())) == sum(
            [len(param_group['params']) for param_group in self.optimizer.param_groups])

    def train(self, n_epochs):

        self.set_seed()

        self.check_consistency()

        for epoch in tqdm(range(n_epochs)):
            self.total_epochs += 1
            train_losses = []
            for batch_x, batch_y in self.train_loader:
                train_losses.append(self.perform_train_step(batch_x, batch_y))
            mini_batch_loss = np.mean(train_losses)
            self.losses.append(mini_batch_loss)

            with torch.no_grad():
                val_losses = []
                for batch_x, batch_y in self.val_loader:
                    val_losses.append(self.perform_val_step(batch_x, batch_y))
                mini_batch_val_loss = np.mean(val_losses)
                self.val_losses.append(mini_batch_val_loss)

            self._epoch_schedulers(mini_batch_val_loss)

            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': mini_batch_loss}
                if mini_batch_val_loss is not None:
                    scalars.update({'validation': mini_batch_val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)

    def predict(self, x, to_numpy=True):
        """
        Passes x though the model to get predictions.

        Args:
            x:

        Returns: self.model(x)

        """

        self.model.eval()
        # need to evaluate
        prediction = self.model(torch.as_tensor(x).float().to(self.device))
        self.model.train()
        if to_numpy:
            return prediction.detach().cpu().numpy()
        else:
            return prediction

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training loss', c='b')
        plt.plot(self.val_losses, label='Validation loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, filename):
        checkpoint = {
            'total_epochs': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'val_losses': self.val_losses,
            }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        """
        From https://pytorch.org/tutorials/beginner/saving_loading_models.html
        "To load the items, first initialize the model and optimizer, then load the dictionary locally using torch.load().
        From here, you can easily access the saved items by simply querying the dictionary as you would expect."

        Args:
            filename:
        """

        checkpoint = torch.load(filename)

        self.total_epochs = checkpoint['total_epochs']

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.losses = checkpoint['losses']
        self.val_losses = checkpoint['val_losses']

        self.model.train()

    def correct(self, x, y, threshold=.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()

        # We get the size of the batch and the number of classes
        # (only 1, if it is binary)
        n_samples, n_dims = yhat.shape
        if n_dims > 1:
            # In a multiclass classification, the biggest logit
            # always wins, so we don't bother getting probabilities

            # This is PyTorch's version of argmax,
            # but it returns a tuple: (max value, index of max value)
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            # In binary classification, if last layer is not Sigmoid we need to apply one:
            if not (isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Sigmoid)):
                yhat = torch.sigmoid(yhat)
            predicted = (yhat > threshold).long()

        # How many samples got classified correctly for each class
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result)

    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        """
        Applies `func` to the loader, batch by batch.

        Args:
            loader:
            func:
            reduce:

        Returns:

        """
        results = [func(x, y) for (x, y) in loader]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)

        return results

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True  # ensures that CUDA selects the same algorithm each time an application is run
        torch.backends.cudnn.benchmark = False  # causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def reset_parameters(self):
        """
        CAUTION this does not reset all parameters like in nn.PReLU for example.
        Based on https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        """
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None,
                           layer_name='', title=None):
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # Sets title, labels, and removes ticks
            if title is not None:
                ax.set_title(f'{title} #{j}', fontsize=12)
            shp = np.atleast_2d(image).shape
            ax.set_ylabel(
                f'{layer_name}\n{shp[0]}x{shp[1]}',
                rotation=0, labelpad=40, fontsize=10
            )
            xlabel1 = '' if y is None else f'\nLabel: {y[j]}'
            xlabel2 = '' if yhat is None else f'\nPredicted: {yhat[j]}'
            xlabel = f'{xlabel1}{xlabel2}'
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plots weight as an image
            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='gray',
                vmin=minv,
                vmax=maxv
            )
        return

    def visualize_filters(self, layer_name, **kwargs):
        try:
            # Gets the layer object from the model
            layer = self.model
            for name in layer_name.split('.'):
                layer = getattr(layer, name)
            # We are only looking at filters for 2D convolutions
            if isinstance(layer, nn.Conv2d):
                # Takes the weight information
                weights = layer.weight.data.cpu().numpy()
                # weights -> (channels_out (filter), channels_in, H, W)
                n_filters, n_channels, _, _ = weights.shape

                # Builds a figure
                size = (2 * n_channels + 2, 2 * n_filters)
                fig, axes = plt.subplots(n_filters, n_channels,
                                         figsize=size)
                axes = np.atleast_2d(axes)
                axes = axes.reshape(n_filters, n_channels)
                # For each channel_out (filter)
                for i in range(n_filters):
                    StepByStep._visualize_tensors(
                        axes[i, :],
                        weights[i],
                        layer_name=f'Filter #{i}',
                        title='Channel'
                    )

                for ax in axes.flat:
                    ax.label_outer()

                fig.tight_layout()
                return fig
            elif isinstance(layer, nn.Linear):  # I added this part for fun
                weights = layer.weight.data.cpu().numpy()
                fig = plt.figure(figsize=(3, 3))
                plt.imshow(weights, cmap='gray')
                plt.grid(False)
                fig.tight_layout()
                return fig
        except AttributeError as e:
            print(e)
            return

    def attach_hooks(self, layers_to_hook, hook_fn=None):
        """
        For hook_fn = None, we'll fill out self.visualization dictionary with layer output.

        Args:
            layers_to_hook:
            hook_fn:

        Returns:

        """

        # Clear any previous values
        self.visualization = {}
        # Creates the dictionary to map layer objects to their names
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]}

        if hook_fn is None:
            # Hook function to be attached to the forward pass
            def hook_fn(layer, inputs, outputs):
                # Gets the layer name
                name = layer_names[layer]
                # Detaches outputs
                values = outputs.detach().cpu().numpy()
                # Since the hook function may be called multiple times
                # for example, if we make predictions for multiple mini-batches
                # it concatenates the results
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([self.visualization[name], values])

        for name, layer in modules:
            # If the layer is in our list
            if name in layers_to_hook:
                # Initializes the corresponding key in the dictionary
                self.visualization[name] = None
                # Register the forward hook and keep the handle in another dict
                self.handles[name] = layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        # Loops through all hooks and removes them
        for handle in self.handles.values():
            handle.remove()
        # Clear the dict, as all hooks have been removed
        self.handles = {}

    def visualize_outputs(self, layers, n_images=10, y=None, yhat=None):
        layers = filter(lambda l: l in self.visualization.keys(), layers)
        layers = list(layers)
        shapes = [self.visualization[layer].shape for layer in layers]
        n_rows = [shape[1] if len(shape) == 4 else 1
                  for shape in shapes]
        total_rows = np.sum(n_rows)

        fig, axes = plt.subplots(total_rows, n_images,
                                 figsize=(1.5 * n_images, 1.5 * total_rows))
        axes = np.atleast_2d(axes).reshape(total_rows, n_images)

        # Loops through the layers, one layer per row of subplots
        row = 0
        for i, layer in enumerate(layers):
            start_row = row
            # Takes the produced feature maps for that layer
            output = self.visualization[layer]

            is_vector = len(output.shape) == 2

            for j in range(n_rows[i]):
                StepByStep._visualize_tensors(
                    axes[row, :],
                    output if is_vector else output[:, j].squeeze(),
                    y,
                    yhat,
                    layer_name=layers[i] \
                        if is_vector \
                        else f'{layers[i]}\nfil#{row - start_row}',
                    title='Image' if (row == 0) else None
                )
                row += 1

        for ax in axes.flat:
            ax.label_outer()

        plt.tight_layout()  # for more space use plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)
        return fig

    @property
    def accuracy_per_class(self):
        return self.loader_apply(self.val_loader, self.correct)

    @property
    def accuracy(self):
        correct, total = self.accuracy_per_class.sum(axis=0)
        accuracy = round(float((correct / total * 100.).cpu().numpy()), 2)
        return accuracy

    @staticmethod
    def statistics_per_channel(images, labels):
        """
        Application is toward DataLoaders, but could be ran on a Dataset as well (torch.unsqueeze adds a dimension).

        Args:
            images (Tensor):
            labels (Tensor):

        Returns:
            Tensor: Size (3, n_channels): [n_samples, sum_means, sum_stds] for example:
                    tensor([[32.0000, 32.0000, 32.0000],
                            [ 6.9379,  7.4181,  6.9894],
                            [ 3.9893,  3.9119,  3.6910]])

        """
        if len(images.shape) < 4:
            # this means there is only one image so n_samples = 1
            images = torch.unsqueeze(images, 0)

        # NCHW
        n_samples, n_channels, n_height, n_weight = images.size()
        # Flatten HW into a single dimension
        flatten_per_channel = images.reshape(n_samples, n_channels, -1)

        # Computes statistics of each image per channel
        # Average pixel value per channel
        # (n_samples, n_channels)
        means = flatten_per_channel.mean(axis=2)
        # Standard deviation of pixel values per channel
        # (n_samples, n_channels)
        stds = flatten_per_channel.std(axis=2)

        # Adds up statistics of all images in a mini-batch
        # (1, n_channels)
        sum_means = means.sum(axis=0)
        sum_stds = stds.sum(axis=0)
        # Makes a tensor of shape (1, n_channels)
        # with the number of samples in the mini-batch
        n_samples = torch.tensor([n_samples] * n_channels).float()

        # Stack the three tensors on top of one another
        # (3, n_channels)
        return torch.stack([n_samples, sum_means, sum_stds])

    @staticmethod
    def make_normalizer(loader):
        """
        Applies statistics_per_channel on a loader that looks for example like this:
                (tensor([6555., 6555., 6555.]),
                 tensor([2109.7141, 2051.4712, 1769.3362]),
                 tensor([541.4397, 492.6389, 484.5458]))
        to get a normalizer.

        Args:
            loader:

        Returns:
            Normalizer

        """

        total_samples, total_means, total_stds = StepByStep.loader_apply(loader, StepByStep.statistics_per_channel)
        norm_mean = total_means / total_samples
        norm_std = total_stds / total_samples
        return Normalize(mean=norm_mean, std=norm_std)

    def lr_range_test(self, data_loader, end_lr, num_iter=100, step_mode='exp', alpha=0.05, ax=None):
        # Since the test updates both model and optimizer we need to store
        # their initial states to restore them in the end
        previous_states = {'model': deepcopy(self.model).state_dict(),
                           'optimizer': deepcopy(self.optimizer).state_dict()}

        # Retrieves the learning rate set in the optimizer
        start_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        # Builds a custom function and corresponding scheduler
        lr_fn = make_lr_fn(start_lr, end_lr, num_iter)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Variables for tracking results and iterations
        tracking = {'loss': [], 'lr': []}
        iteration = 0

        # If there are more iterations than mini-batches in the data loader,
        # it will have to loop over it more than once
        while (iteration < num_iter):
            # That's the typical mini-batch inner loop
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # Step 1
                yhat = self.model(x_batch)
                # Step 2
                loss = self.loss_fn(yhat, y_batch)
                # Step 3
                loss.backward()

                # Here we keep track of the losses (smoothed)
                # and the learning rates
                tracking['lr'].append(scheduler.get_last_lr()[0])
                if iteration == 0:
                    tracking['loss'].append(loss.item())
                else:
                    prev_loss = tracking['loss'][-1]
                    smoothed_loss = alpha * loss.item() + (1 - alpha) * prev_loss
                    tracking['loss'].append(smoothed_loss)

                iteration += 1
                # Number of iterations reached
                if iteration == num_iter:
                    break

                # Step 4
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

        # Restores the original states
        self.optimizer.load_state_dict(previous_states['optimizer'])
        self.model.load_state_dict(previous_states['model'])

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.get_figure()
        ax.plot(tracking['lr'], tracking['loss'])
        if step_mode == 'exp':
            ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        return tracking, fig

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def capture_gradients(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]

        modules = list(self.model.named_modules())
        self._gradients = {}

        def make_log_fn(name, parm_id):
            def log_fn(grad):
                self._gradients[name][parm_id].append(grad.tolist())
                return

            return log_fn

        for name, layer in self.model.named_modules():
            if name in layers_to_hook:
                self._gradients.update({name: {}})
                for parm_id, p in layer.named_parameters():
                    if p.requires_grad:
                        self._gradients[name].update({parm_id: []})
                        log_fn = make_log_fn(name, parm_id)
                        self.handles[f'{name}.{parm_id}.grad'] = p.register_hook(log_fn)
        return

    def capture_parameters(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]

        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules}

        self._parameters = {}

        for name, layer in modules:
            if name in layers_to_hook:
                self._parameters.update({name: {}})
                for parm_id, p in layer.named_parameters():
                    self._parameters[name].update({parm_id: []})

        def fw_hook_fn(layer, inputs, outputs):
            name = layer_names[layer]
            for parm_id, parameter in layer.named_parameters():
                self._parameters[name][parm_id].append(parameter.tolist())

        self.attach_hooks(layers_to_hook, fw_hook_fn)
        return

    def set_lr_scheduler(self, scheduler):
        # Makes sure the scheduler in the argument is assigned to the
        # optimizer we're using in this class
        if scheduler.optimizer == self.optimizer:
            self.scheduler = scheduler
            if (isinstance(scheduler, optim.lr_scheduler.CyclicLR) or
                    isinstance(scheduler, optim.lr_scheduler.OneCycleLR) or
                    isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts)):
                self.is_batch_lr_scheduler = True
            else:
                self.is_batch_lr_scheduler = False

    def _epoch_schedulers(self, val_loss):
        if self.scheduler:
            if not self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)

    def _mini_batch_schedulers(self, frac_epoch):
        if self.scheduler:
            if self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    self.scheduler.step(self.total_epochs + frac_epoch)
                else:
                    self.scheduler.step()

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr)

    def print_trainable_parameters(self):
        print_trainable_parameters(self.model)

    def make_dot(self):
        x, y = next(iter(self.val_loader))

        self.model.eval()
        x = x.to(self.device)
        output = self.model(x)
        dot = make_dot(output)
        dot.render('inception')


def preprocessed_dataset(model, loader, device=None):
    """
    Runs all data in the loader through the model and returns a dataset.
    """

    features = torch.Tensor()
    labels = torch.Tensor().type(torch.long)

    if device is None:
        device = next(model.parameters()).device

    for i, (x_batch, y_batch) in enumerate(loader):
        model.eval()
        output = model(x_batch.to(device))
        features = torch.cat([features, output.detach().cpu()])
        labels = torch.cat([labels, y_batch.cpu()])

    return TensorDataset(features, labels)


def unfreeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def print_trainable_parameters(model):
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if names:
        print("\n".join(names))
    else:
        print('No trainable parameters.')


def compare_optimizers(model, loss_fn, optimizers, train_loader, val_loader=None, schedulers=None, layers_to_hook='', n_epochs=50):
    results = {}
    model_state = deepcopy(model).state_dict()

    for desc, opt in optimizers.items():
        model.load_state_dict(model_state)

        optimizer = opt['class'](model.parameters(), **opt['parms'])

        sbs = StepByStep(model, loss_fn, optimizer)
        sbs.set_loaders(train_loader, val_loader)

        try:
            if schedulers is not None:
                sched = schedulers[desc]
                scheduler = sched['class'](optimizer, **sched['parms'])
                sbs.set_lr_scheduler(scheduler)
        except KeyError:
            pass

        sbs.capture_parameters(layers_to_hook)
        sbs.capture_gradients(layers_to_hook)
        sbs.train(n_epochs)
        sbs.remove_hooks()

        parms = deepcopy(sbs._parameters)
        grads = deepcopy(sbs._gradients)

        lrs = sbs.learning_rates[:]
        if not len(lrs):
            lrs = [list(map(lambda p: p['lr'], optimizer.state_dict()['param_groups']))] * n_epochs

        results.update({desc: {'parms': parms,
                               'grads': grads,
                               'losses': np.array(sbs.losses),
                               'val_losses': np.array(sbs.val_losses),
                               'state': optimizer.state_dict(),
                               'lrs': lrs}})

    return results


class InverseNormalize(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, normalizer):
        mean = normalizer.mean
        std = normalizer.std
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def rescale(x):
    """
    Rescale tensor. Don't have use case yet.

    Args:
        x:

    Returns:

    """
    rescaled_x = torch.clone(x)
    for i in range(x.shape[0]):
        rescaled_x[i,:,:] = rescaled_x[i,:,:] - rescaled_x[i,:,:].min()
        rescaled_x[i,:,:] = rescaled_x[i,:,:] / rescaled_x[i,:,:].max()
    return rescaled_x