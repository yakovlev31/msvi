from abc import ABC, abstractmethod

import torch
import torch.nn as nn


Tensor = torch.Tensor
Module = nn.Module
Parameter = nn.parameter.Parameter


class IDecoder(Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Maps latent state to parameters of p(y|x).

        Args:
            x: Latent state. Has shape (S, M, K).

        Returns:
            param: Parameters of p(y|x). Has shape (S, M, N, D, num. of param. groups in p(y|x)).
                For example, the number of parameter groups in a Normal p(y|x) is 2 (mean and variance).
        """
        pass

    @abstractmethod
    def set_param(self, param: Tensor) -> None:
        """Sets parameters to `param`.

        Args:
            param: New parameter values.
        """
        pass

    @abstractmethod
    def param_count(self) -> int:
        """Calculates the number of parameters.

        Returns:
            The number of parameters.
        """
        pass


class NeuralDecoder(IDecoder):
    """Neural-network-based decoder."""
    def __init__(self, decoder: Module, layers_to_count: list = []) -> None:
        super().__init__()
        self.decoder = decoder
        self.layers_to_count = [
            nn.Linear,
            nn.Conv2d, nn.ConvTranspose2d,
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
        ]  # default
        self.layers_to_count.extend(layers_to_count)  # user-specified (must contain weight and bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def set_param(self, param: Tensor) -> None:
        # Note: after calling set_param() weight and bias of each layer will become tensors,
        #       so calling .parameters() will not show them.
        assert self.param_count() == param.numel(), (
            f"The size of param ({param.numel()}) must be the same as self.param_count()"
            f"({self.param_count()})"
        )
        layers = self._get_layers(self.layers_to_count)
        self._set_layer_param_to_vec(layers, param)

    def param_count(self) -> int:
        param_count = 0
        layers = self._get_layers(self.layers_to_count)
        for layer in layers:
            self._check_weight_and_bias_of_layer(layer)
            layer_param_count = layer.weight.numel() + layer.bias.numel()
            param_count += layer_param_count
        return param_count

    def _get_layers(self, layer_types: list) -> list:
        """Returns all layers in `self.decoder` whose type is present in `layer_types`.

        Args:
            layer_types: A list with the requred layer types (e.g. nn.Linear).

        Returns:
            Layers of `self.decoder` whose type is in `layer_types`.
        """
        return_layers = []
        for layer in self.decoder.modules():
            if type(layer) in layer_types:
                return_layers.append(layer)
        return return_layers

    def _set_layer_param_to_vec(self, layers: list[Module], vec: Tensor) -> None:
        """Sets parameters of Modules in `layers` to elements of `vec`.

        Args:
            layers: List of Modules whose parameters need to be set.
            vec: 1D Tensor with parameters.
        """
        pointer = 0
        for layer in layers:
            self._check_weight_and_bias_of_layer(layer)

            layer_param_count = layer.weight.numel() + layer.bias.numel()  # type: ignore
            layer_weight_count = layer.weight.numel()  # type: ignore

            layer_param = vec[pointer:pointer + layer_param_count]
            layer_weight = layer_param[:layer_weight_count].view_as(layer.weight)  # type: ignore
            layer_bias = layer_param[layer_weight_count:].view_as(layer.bias)  # type: ignore

            self._del_set_layer_attr(layer, "weight", layer_weight)
            self._del_set_layer_attr(layer, "bias", layer_bias)

            pointer += layer_param_count

    def _del_set_layer_attr(self, layer, attr_name, attr_val):
        delattr(layer, attr_name)
        setattr(layer, attr_name, attr_val)

    def _check_weight_and_bias_of_layer(self, layer: Module) -> None:
        assert (type(layer.weight) is Tensor or type(layer.weight) is Parameter), (
            f"weight of layer {layer} must be Tensor or Parameter.")
        assert (type(layer.bias) is Tensor or type(layer.bias) is Parameter), (
            f"bias of layer {layer} must be Tensor or Parameter.")
