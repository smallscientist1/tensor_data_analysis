import torch
import matplotlib.pyplot as plt

def analysis_tensor_data(a: torch.Tensor, plot: bool = False, figure_name: str = 'tensor_distribution.png'):
    print(f"Data type: {a.dtype}")
    print(f"Device: {a.device}")
    print(f"Shape: {a.shape}")
    
    max_value = a.max()
    min_value = a.min()
    mean_value = a.mean()
    std_value = a.std()
    median_value = a.median()
    quantiles = torch.quantile(a, torch.tensor([0.25, 0.5, 0.75]))
    histogram = torch.histc(a, bins=50, min=0, max=1)

    if plot:
        plt.hist(a.cpu().numpy().flatten(), bins=50)
        plt.title('Tensor Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()
        plt.savefig(figure_name)
    
    print(f"Max: {max_value}")
    print(f"Min: {min_value}")
    print(f"Mean: {mean_value}")
    print(f"Std: {std_value}")
    print(f"Median: {median_value}")
    print(f"Quantiles 4: {quantiles}")
    print(f"Histogram: {histogram}")

if __name__ == '__main__':
    a = torch.randn(100, 100)
    analysis_tensor_data(a, plot=True)



