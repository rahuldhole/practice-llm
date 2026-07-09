import torch
import torch.nn as nn

class ModelInspector:
    """
    A non-intrusive debugging helper that registers PyTorch forward and backward
    hooks on specific target layers to capture activation and gradient statistics.
    
    This is extremely useful for:
      - Diagnosing vanishing or exploding gradients.
      - Checking for dead ReLUs (activations always zero).
      - Visualizing model internal behaviors without editing the forward code.
    """
    def __init__(self, model, target_types=(nn.Linear,)):
        """
        Args:
            model (nn.Module): The model to inspect.
            target_types (tuple of nn.Module classes): Module types to attach hooks to.
        """
        self.model = model
        self.target_types = target_types
        self.hook_handles = []
        
        # Repositories for gathered stats
        self.activations = {}
        self.gradients = {}
        
        self._register_hooks()

    def _register_hooks(self):
        """
        Traverses the model's submodule tree and registers hooks on modules matching target_types.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, self.target_types):
                # We need local variables bound to closure for hook functions
                layer_name = name if name != "" else f"{type(module).__name__}_{id(module)}"
                
                # Setup statistics sub-dicts
                self.activations[layer_name] = []
                self.gradients[layer_name] = []
                
                # 1. Register forward hook
                forward_handle = module.register_forward_hook(
                    self._make_forward_hook(layer_name)
                )
                self.hook_handles.append(forward_handle)
                
                # 2. Register full backward hook
                # We use register_full_backward_hook (introduced in modern PyTorch)
                # instead of register_backward_hook which has known issues with multiple inputs/outputs.
                backward_handle = module.register_full_backward_hook(
                    self._make_backward_hook(layer_name)
                )
                self.hook_handles.append(backward_handle)

    def _make_forward_hook(self, layer_name):
        def forward_hook_fn(module, input, output):
            """
            Hook called during the forward pass.
            
            Args:
                module (nn.Module): The layer itself.
                input (tuple of torch.Tensor): Input tensors fed into the layer.
                output (torch.Tensor): Output tensor returned by the layer.
            """
            # Capture stats safely (detach from computational graph to prevent memory leaks)
            with torch.no_grad():
                out_detached = output.detach()
                mean = out_detached.mean().item()
                std = out_detached.std().item()
                shape = list(out_detached.shape)
                
                self.activations[layer_name].append({
                    "mean": mean,
                    "std": std,
                    "shape": shape
                })
        return forward_hook_fn

    def _make_backward_hook(self, layer_name):
        def backward_hook_fn(module, grad_input, grad_output):
            """
            Hook called during the backward pass.
            
            Args:
                module (nn.Module): The layer itself.
                grad_input (tuple of torch.Tensor): Gradients w.r.t the layer's inputs.
                grad_output (tuple of torch.Tensor): Gradients w.r.t the layer's outputs.
            """
            with torch.no_grad():
                # We typically inspect the gradient w.r.t the output of this layer (grad_output[0])
                if grad_output[0] is not None:
                    g_out = grad_output[0].detach()
                    mean = g_out.mean().item()
                    std = g_out.std().item()
                    norm = g_out.norm().item()
                    
                    self.gradients[layer_name].append({
                        "mean": mean,
                        "std": std,
                        "norm": norm
                    })
        return backward_hook_fn

    def remove_hooks(self):
        """
        Removes all registered hooks to avoid memory leaks and overhead.
        """
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        print("[ModelInspector] All hooks removed successfully.")

    def print_summary(self):
        """
        Prints a summary of the captured activation and gradient statistics.
        """
        print(f"\n--- Model Inspector Statistics Summary ---")
        for layer in self.activations.keys():
            print(f"Layer: {layer}")
            
            # Print activation stats
            acts = self.activations[layer]
            if acts:
                latest = acts[-1]
                print(f"  Activations (Latest Step) -> Mean: {latest['mean']:.4f} | Std: {latest['std']:.4f} | Shape: {latest['shape']}")
            else:
                print("  Activations: No runs recorded.")
                
            # Print gradient stats
            grads = self.gradients[layer]
            if grads:
                latest = grads[-1]
                print(f"  Gradients (Latest Step)   -> Mean: {latest['mean']:.4f} | Std: {latest['std']:.4f} | Norm: {latest['norm']:.4f}")
            else:
                print("  Gradients: No backward passes recorded.")


if __name__ == "__main__":
    print("--- PyTorch Forward & Backward Hooks Demo ---")
    
    # 1. Define a simple Multi-Layer Network
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
        nn.Sigmoid()
    )
    # Give custom names to child layers for readable output
    model[0].label = "layer_input_to_hidden"
    model[2].label = "layer_hidden_to_output"
    
    # 2. Attach our inspector to monitor linear layers
    inspector = ModelInspector(model, target_types=(nn.Linear,))
    
    # 3. Simulate Forward and Backward Passes
    x = torch.randn(8, 10)  # batch of 8, features 10
    
    print("\nRunning Forward Pass...")
    out = model(x)
    
    print("Running Backward Pass...")
    loss = out.sum()
    loss.backward()
    
    # 4. Display Collected statistics
    inspector.print_summary()
    
    # 5. Clean up hooks
    inspector.remove_hooks()
    
    # Verify that clean up was successful and subsequent runs don't update hooks
    print("\nRunning another forward and backward pass (should not update inspector)...")
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    # The statistics arrays should not grow longer
    print(f"Recorded forward steps: {len(inspector.activations['0'])}") # layer 0 index
