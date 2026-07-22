import random
import torch
from torch import Tensor
import torch.nn as nn
from itertools import chain


def cfg_drop(data: dict, dropout_prob=0.1) -> dict:
    
    B = len(data["text"])

    for n in range(B):
        
        rand = random.random()
        
        # Full drop
        if rand < dropout_prob:
            update(data, "text", n, "")
            
        # Drop xml block
        # elif p_full <= rand < p_full + p_xml:
        #     pass

        # elif p_full + p_xml <= rand < p_full + p_xml + p_attrib:
        #     pass
        #     p = 0.5
        #     s = drop_xml(data["text"][n], p)
        #     update(data, "text", n, s)

            # if random.random() < p:
            #     update(data, "input_feature", n, 0.)
            #     update(data, "input_mask", n, False)

        # No drop
        else:
            pass

    return data


'''
def cfg_drop(data: dict, p_full=0.1, p_xml=0.1, p_attrib=0.1) -> dict:
    
    B = len(data["text"])

    for n in range(B):
        
        rand = random.random()
        
        # Full drop
        if rand < p_full:
            update(data, "text", n, "")
            
        # Drop xml block
        elif p_full <= rand < p_full + p_xml:
            pass

        elif p_full + p_xml <= rand < p_full + p_xml + p_attrib:
            pass
        #     p = 0.5
        #     s = drop_xml(data["text"][n], p)
        #     update(data, "text", n, s)

            # if random.random() < p:
            #     update(data, "input_feature", n, 0.)
            #     update(data, "input_mask", n, False)

        # No drop
        else:
            pass

    return data
'''

def update(data: dict, key: str, index: int, value) -> None:
    if key in data:
        data[key][index] = value


def drop_xml(s: str, p: float) -> str:
    xml = str_to_xml(s)
    xml = [e for e in xml if random.random() < p]
    return xml_to_str(xml)


def cfg_forward(
    model: nn.Module, 
    t: Tensor, 
    x: Tensor, 
    data_c: dict,  # condition
    data_u: dict,  # uncondition
    cfg_scale=4.0
) -> Tensor:

    x = torch.cat([x, x], dim=0)
    data = cat_list([data_c, data_u], dim=0)
    
    c, u = model(t, x, data).chunk(2, dim=0)
    out = u + cfg_scale * (c - u)

    return out


def cat_list(xs, dim=0):
    out = {}
    for k in xs[0].keys():
        values = [x[k] for x in xs]
        if isinstance(values[0], Tensor):
            out[k] = torch.cat(values, dim=dim)
        else:
            out[k] = list(chain.from_iterable(values))
    return out


if __name__ == '__main__':
    pass