import torch.nn as nn
import torch
from abc import ABC, abstractmethod
import warnings
from typing import Any, List, Optional, Union
import torch.nn.functional as F
import torch.nn.init as init

def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T

class BaseTunerLayer(ABC):
    r"""
    A tuner layer mixin that provides the common methods and attributes for all tuners.

    Args:
        is_plugable (`bool`, *optional*):
            Whether the adapter layer can be plugged to any pytorch module
        active_adapters (Union[List[`str`], `str`], *optional*):
            The name of the active adapter.
    """

    active_adapter = None

    # All names of layers that may contain adapter (trainable) weights
    adapter_layer_names = ()
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ()

    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False

    # the currently active adapter(s)
    _active_adapter = "default"

    # List all merged adapters
    merged_adapters = []

    def get_base_layer(self) -> nn.Module:
        """
        (Recursively) get the base_layer.

        This is necessary for the case that the tuner layer wraps another tuner layer.

        """
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    @property
    def weight(self) -> torch.Tensor:
        # This is required for some transformers code, e.g. for T5, weight is accessed as:
        #     self.wo.weight
        # where "wo" is the adapter layer.
        # https://github.com/huggingface/transformers/blob/78f6ed6c70b29c1560780e3869a7ad4c6b3d2710/src/transformers
        # /models/t5/modeling_t5.py#L292
        base_layer = self.get_base_layer()
        if hasattr(base_layer, "qweight"):
            # QuantLinear
            weight = base_layer.qweight
        else:
            # Other layers
            weight = base_layer.weight
        return weight

    def merge(self, safe_merge: bool = False, task_names = None) -> None:
        raise NotImplementedError

    def unmerge(self) -> None:
        raise NotImplementedError

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_names) -> None:
        """Set the active adapter(s).

        Args:
            adapter_name (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                    # happen if a completely different adapter layer is being activated.
                    layer.requires_grad_(True)
                else:
                    layer.requires_grad_(False)

        self._active_adapter = adapter_names

    def _all_available_adapter_names(self) -> list:
        """Return a sorted list of all available adapter names"""
        adapter_names = set()
        for name in self.adapter_layer_names + self.other_param_names:
            # we check each possible attribute and if it's a dict or ModuleDict, we assume that the keys are the adapter
            # names
            attr = getattr(self, name)
            if hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Delete an adapter from the layer

        This should be called on all adapter layers, or else we will get an inconsistent state.

        This method will also set a new active adapter if the deleted adapter was an active adapter. It is important
        that the new adapter is chosen in a deterministic way, so that the same adapter is chosen on all layers.

        Args:
            adapter_name (`str`): The name of the adapter to delete

        """
        for attr in self.adapter_layer_names + self.other_param_names:
            if adapter_name in getattr(self, attr):
                del getattr(self, attr)[adapter_name]

        if adapter_name in self.active_adapters:
            # choose a new active adapter
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                # no active adapters left, set a new default adapter
                # here we get the list of all adapters existing adapter names and choose the first one
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(remaining_adapters[0])

class LoraLayer(BaseTunerLayer):
    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        # self.lora_embedding_A = nn.ParameterDict({})
        # self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, (nn.Conv1d, nn.ConvTranspose1d)):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, task_name, r, lora_alpha, lora_dropout):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        if isinstance(task_name, list):
            for task_i in task_name:
                self.r[task_i] = r
                self.lora_alpha[task_i] = lora_alpha
                # if lora_dropout > 0.0:
                #     lora_dropout_layer = nn.Dropout(p=lora_dropout)
                # else:
                #     lora_dropout_layer = nn.Identity()

                # self.lora_dropout.update(nn.ModuleDict({task_i: lora_dropout_layer}))
                # Actual trainable parameters
                if r > 0:
                    self.lora_A[task_i] = nn.Linear(self.in_features, r, bias=False)
                    self.lora_B[task_i] = nn.Linear(r, self.out_features, bias=False)
                    init.xavier_normal_(self.lora_A[task_i].weight)
                    init.xavier_normal_(self.lora_B[task_i].weight)
                    self.scaling[task_i] = lora_alpha / r
        elif isinstance(task_name, str):
            self.r[task_name] = r
            self.lora_alpha[task_name] = lora_alpha
            # if lora_dropout > 0.0:
            #     lora_dropout_layer = nn.Dropout(p=lora_dropout)
            # else:
            #     lora_dropout_layer = nn.Identity()
            #
            # self.lora_dropout.update(nn.ModuleDict({task_name: lora_dropout_layer}))
            # Actual trainable parameters
            if r > 0:
                self.lora_A[task_name] = nn.Linear(self.in_features, r, bias=False)
                self.lora_B[task_name] = nn.Linear(r, self.out_features, bias=False)
                init.xavier_normal_(self.lora_A[task_name].weight)
                init.xavier_normal_(self.lora_B[task_name].weight)
                self.scaling[task_name] = lora_alpha / r
        else:
            raise NotImplementedError
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        # self.set_adapter(self.active_adapters)

    def update_layer_conv1d(self, task_name, r, lora_alpha, lora_dropout):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        kernel_size = self.base_layer.kernel_size
        stride = self.base_layer.stride
        padding = self.base_layer.padding
        if isinstance(task_name, list):
            for task_i in task_name:
                self.r[task_i] = r
                self.lora_alpha[task_i] = lora_alpha
                # if lora_dropout > 0.0:
                #     lora_dropout_layer = nn.Dropout(p=lora_dropout)
                # else:
                #     lora_dropout_layer = nn.Identity()

                # self.lora_dropout.update(nn.ModuleDict({task_i: lora_dropout_layer}))
                # Actual trainable parameters
                if r > 0:
                    self.lora_A[task_i] = nn.Conv1d(self.in_features, r, kernel_size, stride, padding, bias=False)
                    self.lora_B[task_i] = nn.Conv1d(r, self.out_features, 1, 1, 0, bias=False)
                    init.xavier_normal_(self.lora_A[task_i].weight)
                    init.xavier_normal_(self.lora_B[task_i].weight)
                    self.scaling[task_i] = lora_alpha / r
        elif isinstance(task_name, str):
            self.r[task_name] = r
            self.lora_alpha[task_name] = lora_alpha
            # if lora_dropout > 0.0:
            #     lora_dropout_layer = nn.Dropout(p=lora_dropout)
            # else:
            #     lora_dropout_layer = nn.Identity()
            #
            # self.lora_dropout.update(nn.ModuleDict({task_name: lora_dropout_layer}))
            # Actual trainable parameters
            if r > 0:
                self.lora_A[task_name] = nn.Conv1d(self.in_features, r, kernel_size, stride, padding, bias=False)
                self.lora_B[task_name] = nn.Conv1d(r, self.out_features, 1, 1, 0, bias=False)
                init.xavier_normal_(self.lora_A[task_name].weight)
                init.xavier_normal_(self.lora_B[task_name].weight)
                self.scaling[task_name] = lora_alpha / r
        else:
            raise NotImplementedError
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        # self.set_adapter(self.active_adapters)

    def update_layer_convtranspose1d(self, task_name, r, lora_alpha, lora_dropout):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        kernel_size = self.base_layer.kernel_size
        stride = self.base_layer.stride
        padding = self.base_layer.padding
        if isinstance(task_name, list):
            for task_i in task_name:
                self.r[task_i] = r
                self.lora_alpha[task_i] = lora_alpha
                if r > 0:
                    self.lora_A[task_i] = nn.ConvTranspose1d(self.in_features, r, kernel_size, stride, padding, bias=False)
                    self.lora_B[task_i] = nn.ConvTranspose1d(r, self.out_features, 1, 1, 0, bias=False)
                    init.xavier_normal_(self.lora_A[task_i].weight)
                    init.xavier_normal_(self.lora_B[task_i].weight)
                    self.scaling[task_i] = lora_alpha / r
        elif isinstance(task_name, str):
            self.r[task_name] = r
            self.lora_alpha[task_name] = lora_alpha
            if r > 0:
                self.lora_A[task_name] = nn.ConvTranspose1d(self.in_features, r, kernel_size, stride, padding, bias=False)
                self.lora_B[task_name] = nn.ConvTranspose1d(r, self.out_features, 1, 1, 0, bias=False)
                init.xavier_normal_(self.lora_A[task_name].weight)
                init.xavier_normal_(self.lora_B[task_name].weight)
                self.scaling[task_name] = lora_alpha / r
        else:
            raise NotImplementedError
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        # self.set_adapter(self.active_adapters)

class LoRALinear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        task_name,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.task_name = task_name

        self.update_layer(task_name, r, lora_alpha, lora_dropout)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, task_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(task_names)}."
            )

        for task_i in task_names:
            if task_i in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(task_name=task_i)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {task_i} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(task_i)
                self.merged_adapters.append(task_i)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            task_name = self.merged_adapters.pop()
            if task_name in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(task_name=task_name)

    def get_delta_weight(self, task_name) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[task_name].weight.device
        dtype = self.lora_B[task_name].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[task_name].weight
        weight_B = self.lora_B[task_name].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[task_name]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[task_name].weight.data = weight_A.to(dtype)
            self.lora_B[task_name].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, task_names: List[str], *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
            for task_i in task_names:
                if task_i not in self.merged_adapters:
                    lora_A = self.lora_A[task_i]
                    lora_B = self.lora_B[task_i]
                    # dropout = self.lora_dropout[task_i]
                    scaling = self.scaling[task_i]
                    x = x.to(lora_A.weight.dtype)
                    result += lora_B(lora_A(x)) * scaling
        else:
            result = self.base_layer(x, *args, **kwargs)
            for task_i in task_names:
                if task_i not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[task_i]
                lora_B = self.lora_B[task_i]
                # dropout = self.lora_dropout[task_i]
                scaling = self.scaling[task_i]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(x)) * scaling
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

class LoRAConv1d(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        task_name,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.task_name = task_name

        self.update_layer_conv1d(task_name, r, lora_alpha, lora_dropout)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, task_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(task_names)}."
            )

        for task_i in task_names:
            if task_i in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(task_name=task_i)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {task_i} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(task_i)
                self.merged_adapters.append(task_i)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            task_name = self.merged_adapters.pop()
            if task_name in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(task_name=task_name)

    def get_delta_weight(self, task_name) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[task_name].weight.device
        dtype = self.lora_B[task_name].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[task_name].weight
        weight_B = self.lora_B[task_name].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        if self.get_base_layer().weight.size()[2] == 1:
            # nn.Conv1d kernel_size=1
            output_tensor = (weight_B.squeeze(2) @ weight_A.squeeze(2)).unsqueeze(2) * self.scaling[task_name]
        else:
            # nn.Conv1d kernel_size>1
            output_tensor = (F.conv1d(weight_A.permute(1, 0, 2), weight_B).permute(1, 0, 2) * self.scaling[task_name])

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            # cast back the weights
            self.lora_A[task_name].weight.data = weight_A.to(dtype)
            self.lora_B[task_name].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, task_names: List[str], *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
            for task_i in task_names:
                if task_i not in self.merged_adapters:
                    lora_A = self.lora_A[task_i]
                    lora_B = self.lora_B[task_i]
                    # dropout = self.lora_dropout[task_i]
                    scaling = self.scaling[task_i]
                    x = x.to(lora_A.weight.dtype)
                    result += lora_B(lora_A(x)) * scaling
        else:
            result = self.base_layer(x, *args, **kwargs)
            for task_i in task_names:
                if task_i not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[task_i]
                lora_B = self.lora_B[task_i]
                # dropout = self.lora_dropout[task_i]
                scaling = self.scaling[task_i]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(x)) * scaling
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

class LoRAConvTranspose1d(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        task_name,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.task_name = task_name

        self.update_layer_convtranspose1d(task_name, r, lora_alpha, lora_dropout)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, task_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(task_names)}."
            )

        for task_i in task_names:
            if task_i in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(task_name=task_i)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {task_i} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(task_i)
                self.merged_adapters.append(task_i)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            task_name = self.merged_adapters.pop()
            if task_name in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(task_name=task_name)

    def get_delta_weight(self, task_name) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[task_name].weight.device
        dtype = self.lora_B[task_name].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[task_name].weight
        weight_B = self.lora_B[task_name].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        if self.get_base_layer().weight.size()[2] == 1:
            # nn.Conv1d kernel_size=1
            output_tensor = (weight_B.squeeze(2) @ weight_A.squeeze(2)).unsqueeze(2)# * self.scaling[task_name]
        else:
            # nn.Conv1d kernel_size>1
            output_tensor = F.conv_transpose1d(weight_A, weight_B) #* self.scaling[task_name]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            # cast back the weights
            self.lora_A[task_name].weight.data = weight_A.to(dtype)
            self.lora_B[task_name].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, task_names: List[str], *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
            for task_i in task_names:
                if task_i not in self.merged_adapters:
                    lora_A = self.lora_A[task_i]
                    lora_B = self.lora_B[task_i]
                    # dropout = self.lora_dropout[task_i]
                    scaling = self.scaling[task_i]
                    x = x.to(lora_A.weight.dtype)
                    result += lora_B(lora_A(x)) * scaling
        else:
            result = self.base_layer(x, *args, **kwargs)
            for task_i in task_names:
                if task_i not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[task_i]
                lora_B = self.lora_B[task_i]
                # dropout = self.lora_dropout[task_i]
                scaling = self.scaling[task_i]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(x)) * scaling
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

if __name__ == "__main__":
    # aa = LoRALinear(nn.Linear(in_features=1000, out_features=2000), task_name=["task10", "task15"], r=30, fan_in_fan_out=False)
    # original_weight = aa.weight.data.clone()
    # print(f"{torch.sum(aa.weight.data.clone() - original_weight)}")
    # aa.merge(task_names=["task10"])
    # aa.merge(task_names=["task15"])
    # # print(f"{torch.sum(aa.weight.data.clone() - original_weight)}")
    # # result = torch.sum(aa.weight.data.clone() - original_weight - aa.get_delta_weight(task_name="task10"))
    # # print(result)
    # aa.unmerge()
    # print(f"{torch.sum(aa.weight.data.clone() - original_weight)}")
    # print("===================")

    # bb = LoRAConv1d(nn.Conv1d(in_channels=50, out_channels=40, kernel_size=3, padding=1), task_name=["task10", "task15"], r=30, fan_in_fan_out=False)
    # original_weight = bb.weight.data.clone()
    # print(f"{torch.sum(bb.weight.data.clone() - original_weight)}")
    # bb.merge(task_names=["task10"])
    # bb.merge(task_names=["task15"])
    # # print(f"{torch.sum(aa.weight.data.clone() - original_weight)}")
    # # result = torch.sum(aa.weight.data.clone() - original_weight - aa.get_delta_weight(task_name="task10"))
    # # print(result)
    # bb.unmerge()
    # print(f"{torch.sum(bb.weight.data.clone() - original_weight)}")
    # print("===================")

    cc = LoRAConvTranspose1d(nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), task_name=["task10", "task15"], r=32, fan_in_fan_out=False)
    original_weight = cc.weight.data.clone()
    print(f"{torch.sum(cc.weight.data.clone() - original_weight)}")
    cc.merge(task_names=["task10"])
    cc.merge(task_names=["task15"])
    cc.unmerge()
    print(f"{torch.sum(cc.weight.data.clone() - original_weight)}")
    print("===================")