
import torch
import torch.nn as nn
from plotting import plot_prediction
import traceback

# Dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        # Flatten input: batch_size x 3 x 10 x 10 -> batch_size x 300
        # Wait, the model expects something else?
        # The real model expects 3x128x128.
        # My dummy loader yields 3x10x10.
        # My dummy model expects input size 10?
        # x is 2x3x10x10.
        # I should probably flatten it or change model input.
        # Let's just make the model accept whatever shape and return 2 outputs.
        return torch.randn(x.size(0), 2)

# Dummy loader
class DummyLoader:
    def __iter__(self):
        # Yield a batch of images (batch_size=2, channels=3, height=10, width=10)
        # and labels
        yield torch.randn(2, 3, 10, 10), torch.tensor([0, 1])

# Test
try:
    model = SimpleModel()
    loader = DummyLoader()
    class_names = ['Class0', 'Class1']
    
    # We mock matplotlib.pyplot.show to avoid blocking
    import matplotlib.pyplot as plt
    plt.show = lambda: None
    
    print("Running plot_prediction...")
    plot_prediction(model, loader, class_names, device='cpu', columns=1, rows=1)
    print("plot_prediction ran successfully")
except Exception:
    traceback.print_exc()
