import sys
import unittest
from unittest.mock import MagicMock, patch

class TestFastVAEHook(unittest.TestCase):
    def setUp(self):
        # Mock torch and comfy
        self.mock_torch = MagicMock()
        sys.modules['torch'] = self.mock_torch
        sys.modules['torch.nn'] = MagicMock()
        sys.modules['torch.nn.functional'] = MagicMock()

    def tearDown(self):
        # Clean up mocks
        del sys.modules['torch']
        del sys.modules['torch.nn']
        del sys.modules['torch.nn.functional']

    def test_hook_loads(self):
        # Add mock for internal modules
        sys.modules['src.models.video_vae_v3.modules.causal_inflation_lib'] = MagicMock()
        sys.modules['src.models.video_vae_v3.modules.types'] = MagicMock()

        from src.optimization.fast_vae_hook import apply_fast_vae_patch
        apply_fast_vae_patch(enable=True)

if __name__ == '__main__':
    unittest.main()
