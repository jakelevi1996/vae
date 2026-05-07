import torch
import juml

class MultiHeadReluMlp(juml.base.Model):
    def __init__(
        self,
        input_dim:          int,
        output_dims:        list[int],
        hidden_dim:         int,
        num_hidden_layers:  int,
    ):
        self._torch_module_init()
        self.hidden_layers = torch.nn.ModuleList()
        self.output_layers = torch.nn.ModuleList()
        layer_input_dim = input_dim

        for _ in range(num_hidden_layers):
            layer = juml.models.Linear(layer_input_dim, hidden_dim)
            self.hidden_layers.append(layer)
            layer_input_dim = hidden_dim

        for d in output_dims:
            layer = juml.models.Linear(layer_input_dim, d)
            self.output_layers.append(layer)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        for layer in self.hidden_layers:
            x = layer.forward(x)
            x = torch.relu(x)

        y_list = [
            layer.forward(x)
            for layer in self.output_layers
        ]

        return y_list
