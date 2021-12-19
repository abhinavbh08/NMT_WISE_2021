import torch
from torch._C import dtype
import torch.nn as nn

def sequence_mask(x, valid_len, value=0):

    maxlen = x.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x

# x = torch.tensor([[7, 2, 3], [4, 5, 6]])
# print(sequence_mask(x, torch.tensor([1, 2])))

# y = x.masked_fill(x == 1, 0)
# print(y)

class MaskedCELoss(nn.CrossEntropyLoss):

    def forward(self, pred, labels, valid_len):
        weights = torch.ones_like(labels)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        raw_loss = super(MaskedCELoss, self).forward(pred.permute(0, 2, 1), labels)
        weighted_loss = (raw_loss * weights).mean(dim=1)
        return weighted_loss


# loss = MaskedCELoss()
# l = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
#      torch.tensor([4, 2, 0]))
# print(l)

# pred = torch.ones(3, 4, 10)
# labels = torch.Tensor([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]]).to(torch.long)

# loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="none")(pred.permute(0, 2, 1), labels)
# print(loss.mean(dim=1))

# torch.nn.CrossEntropyLoss