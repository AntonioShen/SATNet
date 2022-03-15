import torch

from tqdm import tqdm
from dataloader import get_data


@torch.no_grad()
def get_mapping(model, device, params, train=True, num_cats=9):
    confusion = torch.zeros([num_cats, 9])
    dataloader = get_data(params['dataset'], params['batch_size'], train=train)
    tloader = tqdm(dataloader)
    for batch_idx, (x, y) in enumerate(tloader):
        x = x.to(device)
        y = y.to(device)

        c1, c2, c3 = model(x)

        exp_c1 = torch.exp(c1)
        probs = exp_c1/torch.sum(exp_c1, dim=1, keepdim=True)

        values, indices = torch.topk(probs, 1)
        indices = indices.squeeze()


        for i, index in enumerate(indices):
            confusion[index, y[i] - 1] += 1

        tloader.set_description(f'batch {batch_idx}')

    values, indices = torch.topk(confusion, 1)

    # zyy
    # print(confusion.numpy())

    return values, indices.squeeze(), len(dataloader.dataset)