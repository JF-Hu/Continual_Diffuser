import torch
import numpy as np
import random


DEVICE = 'cuda'
DTYPE = torch.float

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def seed_configuration(argus):
	if argus.reset_seed:
		argus.seed = np.random.randint(0, 999)
	set_seed(argus.seed)
	return argus

def to_device(x, device=DEVICE, convert_to_torch_float=False):
	if torch.is_tensor(x):
		if convert_to_torch_float:
			x = x.type(torch.FloatTensor)
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v, device) for k, v in x.items()}
	else:
		raise Exception(f'Unrecognized type in `to_device`: {type(x)}')

def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
		# import pdb; pdb.set_trace()
	return torch.tensor(x, dtype=dtype, device=device)

def batch_to_device(batch, device='cuda:0', convert_to_torch_float=False):
	vals = [
		to_device(getattr(batch, field), device, convert_to_torch_float)
		for field in batch._fields
	]
	return type(batch)(*vals)

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def apply_dict(fn, d, *args, **kwargs):
	return {
		k: fn(v, *args, **kwargs)
		for k, v in d.items()
	}

def batchify(batch, device):
	'''
		convert a single dataset item to a batch suitable for passing to a model by
			1) converting np arrays to torch tensors and
			2) and ensuring that everything has a batch dimension
	'''
	fn = lambda x: to_torch(x[None], device=device)

	batched_vals = []
	for field in batch._fields:
		val = getattr(batch, field)
		val = apply_dict(fn, val) if type(val) is dict else fn(val)
		batched_vals.append(val)
	return type(batch)(*batched_vals)






