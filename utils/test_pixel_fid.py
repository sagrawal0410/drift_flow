import unittest
import os
import shutil
import torch
from torchvision.utils import save_image
from nn_flow.utils.pixel_fid import mnist_pixel_fid

class TestPixelFid(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_pixel_fid_images"
        os.makedirs(self.test_dir, exist_ok=True)
        # Create some dummy images
        for i in range(10):
            img = torch.rand(1, 28, 28)
            save_image(img, os.path.join(self.test_dir, f"img_{i}.png"))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_mnist_pixel_fid(self):
        # We expect the FID to be a non-negative number.
        # A more thorough test would require known statistics and a fixed set of images.
        # For now, we just check if it runs and produces a valid score.
        fid_score = mnist_pixel_fid(self.test_dir)
        self.assertIsInstance(fid_score, torch.Tensor)
        self.assertGreaterEqual(fid_score.item(), 0)

if __name__ == '__main__':
    unittest.main() 