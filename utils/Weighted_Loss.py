import torch


class weighted_loss():
    def __init__(self, num_labels, batch_size, device, criterion=None):
        '''
        Function Description:
            Init
        Parameters:
            num_labels: Numbers of true labels
            batch_size: batch size
            device: device
            criterion: Needed if you are using GHM loss
                Example: criterion = nn.CrossEntropy()
        '''
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.device = device
        self.criterion = criterion
        self.eps = 1e-4

    def compute_total(self, labels):
        total = 0
        for i in range(self.num_labels):
            num_i = torch.numel(labels[labels == i])
            if num_i:
                total += 1
        return (total - 1) * self.batch_size + self.eps

    def balanced_cross_entropy(self, label_out, labels):
        '''
        Function Description:
            Balanced Cross Entropy
        Parameters:
            label_out: Model output with shape [batch_size, num_labels]
            labels: True labels with shape [batch_size, 1]
        '''
        loss = torch.tensor(0.).to(self.device)
        weights = (torch.zeros_like(label_out[0])).float()
        total = self.compute_total(labels)
        for i in range(self.num_labels):
            temp_label_out = label_out[labels == i]
            weights[i] = (self.eps + label_out.shape[0] - torch.numel(labels[labels == i])) / total
            if temp_label_out.numel() > 0:
                loss += torch.sum(- weights[i] * torch.log(temp_label_out[:, i]))

        return loss / self.batch_size

    def focal_loss(self, label_out, labels, r):
        '''
        Function Description:
            Focal Loss
        Parameters:
            label_out: Model output with shape [batch_size, num_labels]
            labels: True labels with shape [batch_size, 1]
            r: Exponent
        '''
        loss = torch.tensor(0.).to(self.device)
        weights = (torch.zeros_like(label_out[0])).float()
        total = self.compute_total(labels)
        for i in range(self.num_labels):
            temp_label_out = label_out[labels == i]
            weights[i] = (self.eps + label_out.shape[0] - torch.numel(labels[labels == i])) / total
            if temp_label_out.numel() > 0:
                loss += torch.sum(- weights[i] * ((1 - temp_label_out[:, i]) ** r) * torch.log(temp_label_out[:, i]))

        return loss / self.batch_size

    def ghm_loss(self, label_out, labels, num_intervals):
        '''
        Function Description:
            GHM Loss
        Parameters:
            label_out: Model output with shape [batch_size, num_labels]
            labels: True labels with shape [batch_size, 1]
            num_intervals: Numbers of intervals of grad norm
            reduction: Loss reduction
        '''
        axis = torch.arange(num_intervals + 1).float() / (num_intervals // self.num_labels - 1)
        weights = (torch.zeros_like(labels)).float()
        pred = torch.max(label_out, dim=1)[1]
        g = torch.abs(pred.detach() - labels)

        for i in range(num_intervals):
            inds = (g >= axis[i]) & (g < axis[i + 1])
            nums = inds.sum().item()
            if nums:
                weights[inds] = labels.shape[0] / nums

        loss = torch.tensor(0.).to(self.device)

        for i in range(labels.shape[0]):
            loss += weights[i] * self.criterion(label_out[i].unsqueeze(0), labels[i].unsqueeze(0))

        return loss / self.batch_size