import numpy as np
import torch
from utils.misc import EasyDict
import functools
from utils.logging_utils import plt_to_image
import matplotlib.pyplot as plt
'''
2D Toy Datasets. 
All datasets support function: 
    sample(batch_size) -> [batch_size, 2] tensor
    usually the data is in [-3, 3] x [-3, 3]

Suppors a function eval(dataset, generator=None, n_samples=2048) -> dict:
'''
def convert_to_tensor(data, device="cuda"):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, device=device)
    else:
        raise ValueError(f"Unknown data type: {type(data)}")
def nn_distance_dict(data, queries, device="cuda"):
    """
    Returns a dictionary contains:
        "nn_data": nearest neighbor distance within data
        "nn_precision": nearest neighbor distance from queries to data
        "nn_recall": nearest neighbor distance from data to queries
    """
    data = convert_to_tensor(data, device)
    queries = convert_to_tensor(queries, device)
    nn_data = torch.cdist(data, data)
    # Fill diagonal with inf to exclude self-distances
    nn_data.fill_diagonal_(float("inf"))
    nn_data = nn_data.min(dim=1).values.mean()

    # Compute distances between queries and data points
    nn_precision = torch.cdist(queries, data).min(dim=1).values.mean()
    nn_recall = torch.cdist(data, queries).min(dim=1).values.mean()

    return {"nn_data": nn_data, "nn_precision": nn_precision, "nn_recall": nn_recall}



class SpiralDataset:
    def __init__(
        self,
        n_arms=2,
        noise=0.1,
        start_radius=0,
        end_radius=1,
        rotations=2,
        device="cuda",
    ):
        """
        Args:
            n_arms: Number of spiral arms
            noise: Standard deviation of Gaussian noise added to points
            start_radius: Starting radius of spiral
            end_radius: Ending radius of spiral
            rotations: Number of complete rotations from start to end
            device: Device to place tensors on
        """
        self.n_arms = n_arms
        self.noise = noise
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.rotations = rotations
        self.device = device

    def sample(self, batch_size):
        # Ensure equal points per arm
        points_per_arm = batch_size // self.n_arms
        remainder = batch_size % self.n_arms

        samples = []
        for i in range(self.n_arms):
            # Add extra point to some arms if batch_size isn't divisible by n_arms
            current_n = points_per_arm + (1 if i < remainder else 0)

            # Generate radius and angle parameters
            r = np.linspace(self.start_radius, self.end_radius, current_n)
            theta = 2 * np.pi * self.rotations * r + (2 * np.pi * i / self.n_arms)

            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # Add noise
            if self.noise > 0:
                x += np.random.normal(0, self.noise, current_n)
                y += np.random.normal(0, self.noise, current_n)

            spiral_arm = np.stack([x, y], axis=1)
            samples.append(spiral_arm)

        # Combine all spiral arms
        samples = np.concatenate(samples, axis=0)
        return torch.FloatTensor(samples).to(self.device)

def spiral(**override_kwargs):
    kwargs = {
        "n_arms": 2,
        "noise": 0.02,
        "start_radius": 0.2,
        "end_radius": 2.0,
        "rotations": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    for k, v in override_kwargs.items():
        kwargs[k] = v
    return SpiralDataset(**kwargs)

def four_gaussians(std=0.15, device="cuda"):
    centers = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    stds = [std] * 4
    phi = [1.0] * 4
    mu = centers
    Sigma = [std**2 * np.eye(2) for std in stds]
    return GaussianMixture(phi, mu, Sigma).to(device)


def eight_gaussians(std=0.3, device="cuda"):
    centers = [(1, 1), (-1, 1), (-1, -1), (1, -1), (0, 2), (0, -2), (2, 0), (-2, 0)]
    stds = [std] * len(centers)
    phi = [1.0] * len(centers)
    mu = centers
    Sigma = [std**2 * np.eye(2) for std in stds]
    return GaussianMixture(phi, mu, Sigma).to(device)


def biased_mixtures(std=0.3, device="cuda", weights=[1.0, 0.5, 0.5], centers=[(1, 1), (-1, 1), (-1, -1)]):
    stds = [std] * len(centers)
    Sigma = [std**2 * np.eye(2) for std in stds]
    return GaussianMixture(weights, centers, Sigma).to(device)

class GaussianMixture(torch.nn.Module):
    def __init__(
        self,
        phi,  # Per-component weight: [comp]
        mu,  # Per-component mean: [comp, dim]
        Sigma,  # Per-component covariance matrix: [comp, dim, dim]
        sample_lut_size=64 << 10,  # Lookup table size for efficient sampling.
    ):
        super().__init__()
        self.register_buffer(
            "phi", torch.tensor(np.asarray(phi) / np.sum(phi), dtype=torch.float32)
        )
        self.register_buffer("mu", torch.tensor(np.asarray(mu), dtype=torch.float32))
        self.register_buffer(
            "Sigma", torch.tensor(np.asarray(Sigma), dtype=torch.float32)
        )

        # Precompute eigendecompositions of Sigma for efficient heat diffusion.
        L, Q = torch.linalg.eigh(self.Sigma)  # Sigma = Q @ L @ Q
        self.register_buffer("_L", L)  # L: [comp, dim, dim]
        self.register_buffer("_Q", Q)  # Q: [comp, dim, dim]

        # Precompute lookup table for efficient sampling.
        self.register_buffer(
            "_sample_lut", torch.zeros(sample_lut_size, dtype=torch.int64)
        )
        phi_ranges = (
            torch.cat([torch.zeros_like(self.phi[:1]), self.phi.cumsum(0)])
            * sample_lut_size
            + 0.5
        ).to(torch.int32)
        for idx, (begin, end) in enumerate(zip(phi_ranges[:-1], phi_ranges[1:])):
            self._sample_lut[begin:end] = idx

    # Evaluate the terms needed for calculating PDF and score.
    def _eval(self, x, sigma=0):  # x: [..., dim], sigma: [...]
        L = self._L + sigma[..., None, None] ** 2  # L' = L + sigma * I: [..., dim]
        d = L.prod(-1)  # d = det(Sigma') = det(Q @ L' @ Q) = det(L'): [...]
        y = self.mu - x[..., None, :]  # y = mu - x: [..., comp, dim]
        z = torch.einsum(
            "...ij,...j,...kj,...k->...i", self._Q, 1 / L, self._Q, y
        )  # z = inv(Sigma') @ (mu - x): [..., comp, dim]
        c = (
            self.phi / (((2 * np.pi) ** x.shape[-1]) * d).sqrt()
        )  # normalization factor of N(x; mu, Sigma')
        w = (
            c * (-1 / 2 * torch.einsum("...i,...i->...", y, z)).exp()
        )  # w = N(x; mu, Sigma'): [..., comp]
        return z, w

    # Calculate p(x; sigma) for the given sample points, processing at most the given number of samples at a time.
    def pdf(self, x, sigma=0, max_batch_size=1 << 14):
        sigma = torch.as_tensor(
            sigma, dtype=torch.float32, device=x.device
        ).broadcast_to(x.shape[:-1])
        x_batches = x.flatten(0, -2).split(max_batch_size)
        sigma_batches = sigma.flatten().split(max_batch_size)
        pdf_batches = [
            self._eval(xx, ss)[1].sum(-1) for xx, ss in zip(x_batches, sigma_batches)
        ]
        return torch.cat(pdf_batches).reshape(x.shape[:-1])  # x.shape[:-1]

    # Calculate log(p(x; sigma)) for the given sample points, processing at most the given number of samples at a time.
    def logp(self, x, sigma=0, max_batch_size=1 << 14):
        return self.pdf(x, sigma, max_batch_size).log()

    # Calculate \nabla_x log(p(x; sigma)) for the given sample points.
    def score(self, x, sigma=0):
        sigma = torch.as_tensor(
            sigma, dtype=torch.float32, device=x.device
        ).broadcast_to(x.shape[:-1])
        z, w = self._eval(x, sigma)
        w = w[..., None]
        return (w * z).sum(-2) / w.sum(-2)  # x.shape

    # Draw the given number of random samples from p(x; sigma).
    def sample(self, shape, sigma=0, generator=None):
        sigma = torch.as_tensor(
            sigma, dtype=torch.float32, device=self.mu.device
        ).broadcast_to(shape)
        i = self._sample_lut[
            torch.randint(
                len(self._sample_lut),
                size=sigma.shape,
                device=sigma.device,
                generator=generator,
            )
        ]
        L = self._L[i] + sigma[..., None] ** 2  # L' = L + sigma * I: [..., dim]
        x = torch.randn(
            L.shape, device=sigma.device, generator=generator
        )  # x ~ N(0, I): [..., dim]
        y = torch.einsum(
            "...ij,...j,...kj,...k->...i", self._Q[i], L.sqrt(), self._Q[i], x
        )  # y = sqrt(Sigma') @ x: [..., dim]
        return y + self.mu[i]  # [..., dim]


# ----------------------------------------------------------------------------
# Construct a ground truth 2D distribution for the given set of classes
# ('A', 'B', or 'AB').



@functools.lru_cache(None)
def edm_toy(
    classes="A",
    device=torch.device("cpu"),
    seed=2,
    origin=np.array([0.0030, 0.0325]),
    scale=np.array([1.3136, 1.3844]),
):
    rnd = np.random.RandomState(seed)
    comps = []

    # Recursive function to generate a given branch of the distribution.
    def recurse(cls, depth, pos, angle):
        if depth >= 7:
            return

        # Choose parameters for the current branch.
        dir = np.array([np.cos(angle), np.sin(angle)])
        dist = 0.292 * (0.8**depth) * (rnd.randn() * 0.2 + 1)
        thick = 0.2 * (0.8**depth) / dist
        size = scale * dist * 0.06

        # Represent the current branch as a sequence of Gaussian components.
        for t in np.linspace(0.07, 0.93, num=8):
            c = EasyDict()
            c.cls = cls
            c.phi = dist * (0.5**depth)
            c.mu = (pos + dir * dist * t) * scale
            c.Sigma = (
                np.outer(dir, dir) + (np.eye(2) - np.outer(dir, dir)) * (thick**2)
            ) * np.outer(size, size)
            comps.append(c)

        # Generate each child branch.
        for sign in [1, -1]:
            recurse(
                cls=cls,
                depth=(depth + 1),
                pos=(pos + dir * dist),
                angle=(angle + sign * (0.7**depth) * (rnd.randn() * 0.2 + 1)),
            )

    # Generate each class.
    recurse(cls="A", depth=0, pos=origin, angle=(np.pi * 0.25))
    recurse(cls="B", depth=0, pos=origin, angle=(np.pi * 1.25))

    # Construct a GaussianMixture object for the selected classes.
    sel = [c for c in comps if c.cls in classes]
    distrib = GaussianMixture(
        [c.phi for c in sel], [c.mu for c in sel], [c.Sigma for c in sel]
    )
    return distrib.to(device)

def draw_quadrants(data, gen, n_samples):
    """
    Count points in each quadrant and generate a heatmap visualization.
    
    Args:
        data: tensor or array of shape [n_samples, 2]
        gen: tensor or array of shape [n_samples, 2]
        n_samples: number of samples
        
    Returns:
        quadrant_fig: heatmap visualization
        quadrant_counts: dictionary with counts in each quadrant
    """
    # Convert to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(gen, torch.Tensor):
        gen = gen.detach().cpu().numpy()
    
    # Count points in each quadrant
    data_q1 = np.sum((data[:, 0] >= 0) & (data[:, 1] >= 0))
    data_q2 = np.sum((data[:, 0] < 0) & (data[:, 1] >= 0))
    data_q3 = np.sum((data[:, 0] < 0) & (data[:, 1] < 0))
    data_q4 = np.sum((data[:, 0] >= 0) & (data[:, 1] < 0))
    
    gen_q1 = np.sum((gen[:, 0] >= 0) & (gen[:, 1] >= 0))
    gen_q2 = np.sum((gen[:, 0] < 0) & (gen[:, 1] >= 0))
    gen_q3 = np.sum((gen[:, 0] < 0) & (gen[:, 1] < 0))
    gen_q4 = np.sum((gen[:, 0] >= 0) & (gen[:, 1] < 0))
    
    # Calculate fractions
    data_fractions = np.array([
        [data_q3/n_samples, data_q2/n_samples],  # Bottom left, Top left
        [data_q4/n_samples, data_q1/n_samples]   # Bottom right, Top right
    ])
    
    gen_fractions = np.array([
        [gen_q3/n_samples, gen_q2/n_samples],  # Bottom left, Top left
        [gen_q4/n_samples, gen_q1/n_samples]   # Bottom right, Top right
    ])

    quadrant_counts = {
        'data': {
            "Q1 (+,+)": int(data_q1),
            "Q2 (-,+)": int(data_q2),
            "Q3 (-,-)": int(data_q3),
            "Q4 (+,-)": int(data_q4)
        },
        'generated': {
            "Q1 (+,+)": int(gen_q1),
            "Q2 (-,+)": int(gen_q2),
            "Q3 (-,-)": int(gen_q3),
            "Q4 (+,-)": int(gen_q4)
        }
    }
    
    # Create heatmap visualization of quadrant distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Increased figure width
    
    # Data heatmap
    im1 = ax1.imshow(data_fractions, cmap='viridis', vmin=0, vmax=max(data_fractions.max(), gen_fractions.max()))
    ax1.set_title('Data Distribution')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['x < 0', 'x ≥ 0'])
    ax1.set_yticklabels(['y < 0', 'y ≥ 0'])
    
    # Add percentage text to each cell
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{data_fractions[i, j]:.1%}", 
                    ha="center", va="center", color="w" if data_fractions[i, j] > 0.3 else "black")
    
    # Generated data heatmap
    im2 = ax2.imshow(gen_fractions, cmap='viridis', vmin=0, vmax=max(data_fractions.max(), gen_fractions.max()))
    ax2.set_title('Generated Distribution')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['x < 0', 'x ≥ 0'])
    ax2.set_yticklabels(['y < 0', 'y ≥ 0'])
    
    # Add percentage text to each cell
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{gen_fractions[i, j]:.1%}", 
                    ha="center", va="center", color="w" if gen_fractions[i, j] > 0.3 else "black")
    
    # Add colorbar with adjusted parameters
    # cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.05)
    # cbar.set_label('Fraction of Points')
    
    plt.tight_layout(pad=3.0)
    quadrant_fig = plt_to_image()
    plt.close()
    
    return quadrant_fig, quadrant_counts

def eval(dataset, generator=None, n_samples=2048):
    '''
    Evaluate the generator's performance. 
    Args:
        generator: a function that takes a batch_size and returns a tensor of shape [batch_size, 2]
        dataset: a dataset object that supports sample(batch_size) -> [batch_size, 2] tensor
    Returns:
        info: a dictionary containing the following keys:
            'nn_data': nearest neighbor distance
            'precision': precision
            'recall': recall
            'fig': an image of the dataset and the generator's samples
            'hist_fig': histogram of precision and recall metrics
            'quadrant_fig': heatmap showing point distribution across quadrants
    '''
    with torch.no_grad():   
        data = dataset.sample(n_samples)
        gen = generator(n_samples)
        if not isinstance(gen, torch.Tensor):
            gen = torch.tensor(gen, device=data.device)  
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, device=gen.device)
        gen = gen.to(data.device)

        gen = gen.clamp(-3, 3)
        data = data.clamp(-3, 3)
        cdist = torch.cdist(data, gen)
        info = nn_distance_dict(data, gen)

    data_np = data.detach().cpu().numpy()
    gen_np = gen.detach().cpu().numpy()

    # Create scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(data_np[:, 0], data_np[:, 1], label='data', s=5, alpha=0.2,  color='blue')
    plt.scatter(gen_np[:, 0], gen_np[:, 1], label='gen', s=5, alpha=0.2, color='red')
    plt.legend()
    plt.tight_layout()
    scatter_fig = plt_to_image()
    plt.close()
    
    # Create histogram plot with aligned axes
    plt.figure(figsize=(8, 4))
    precision = cdist.min(dim=0).values.cpu().numpy()
    recall = cdist.min(dim=1).values.cpu().numpy()
    
    # Find common min and max for both histograms
    min_val = min(precision.min(), recall.min())
    max_val = max(precision.max(), recall.max())
    bins = np.linspace(min_val, max_val, 30)
    
    plt.hist(precision, bins=bins, alpha=0.5, label='precision', color='green')
    plt.hist(recall, bins=bins, alpha=0.5, label='recall', color='orange')
    plt.legend()
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Precision and Recall Histograms')
    plt.tight_layout()
    hist_fig = plt_to_image()
    plt.close()
    
    # Draw quadrant heatmap
    quadrant_fig, quadrant_counts = draw_quadrants(data_np, gen_np, n_samples)
    
    return {
        'nn_data': info['nn_data'],
        'precision': info['nn_precision'],
        'recall': info['nn_recall'],
        'fig': scatter_fig,
        'hist_fig': hist_fig,
        'quadrant_fig': quadrant_fig,
    }

def build_dataset(name, **kwargs):
    '''
    Build a dataset object. 
    Args:
        dataset_name: the name of the dataset; "four_gaussians", "edm_toy", "spiral", "eight_gaussians", "biased_mixtures"
    Returns:
        dataset: a dataset object that supports sample(batch_size) -> [batch_size, 2] tensor
    '''
    assert name in ["four_gaussians", "edm_toy", "spiral", "eight_gaussians", "biased_mixtures"]
    map_name = {
        "four_gaussians": four_gaussians,
        "eight_gaussians": eight_gaussians,
        "biased_mixtures": biased_mixtures,
        "edm_toy": edm_toy,
        "spiral": spiral,
    }
    if name not in map_name:
        raise ValueError(f"Dataset {name} not found")
    return map_name[name](**kwargs)

if __name__ == "__main__":
    dataset = [four_gaussians(), edm_toy(), spiral(), eight_gaussians(), biased_mixtures()][-1]
    res = eval(dataset, dataset.sample)
    print(res['nn_data'], res['precision'], res['recall'])
    fig = res['fig']
    hist_fig = res['hist_fig']
    quadrant_fig = res['quadrant_fig']
    # Save the image objects properly
    fig.save('fig.png')
    hist_fig.save('hist_fig.png')
    quadrant_fig.save('quadrant_fig.png')
    fig.show()
    hist_fig.show()
    quadrant_fig.show()
# %%
