import torch
import torch.nn as nn

class ModelInspector:
    """
    A debugging camera helper that attaches forward and backward hooks
    to modules to monitor activations and gradients without altering code.
    """
    def __init__(self, model, target_type=nn.Linear):
        self.model = model
        self.target_type = target_type
        self.hook_handles = []
        
        # Repositories to store stats
        self.activations = {}
        self.gradients = {}
        
        # Automatically attach hooks
        self._attach_hooks()

    def _attach_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, self.target_type):
                # Unique label for this layer
                label = name if name != "" else f"layer_{id(module)}"
                self.activations[label] = []
                self.gradients[label] = []
                
                # 1. Attach forward hook (snaps activation statistics)
                f_handle = module.register_forward_hook(self._make_forward_hook(label))
                self.hook_handles.append(f_handle)
                
                # 2. Attach backward hook (snaps gradient statistics)
                b_handle = module.register_full_backward_hook(self._make_backward_hook(label))
                self.hook_handles.append(b_handle)

    def _make_forward_hook(self, label):
        def hook_fn(module, input, output):
            # Detach output tensor to prevent memory leaks in autograd
            out_detached = output.detach()
            self.activations[label].append({
                "mean": out_detached.mean().item(),
                "shape": list(out_detached.shape)
            })
        return hook_fn

    def _make_backward_hook(self, label):
        def hook_fn(module, grad_input, grad_output):
            if grad_output[0] is not None:
                g_detached = grad_output[0].detach()
                self.gradients[label].append({
                    "mean": g_detached.mean().item(),
                    "norm": g_detached.norm().item()
                })
        return hook_fn

    def remove(self):
        # Detach all hooks from the model layers
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
