from torch.utils.data._utils.collate import default_collate_fn_map

def _collate_list_fn(batch, *, collate_fn_map=None):
    return batch

default_collate_fn_map.update({list: _collate_list_fn})