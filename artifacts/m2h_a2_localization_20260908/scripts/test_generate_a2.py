import unittest
from generate_a2 import check_state


class Value:
    def __init__(self, shape): self.shape = shape


class CompatibilityTest(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(check_state({'a':Value((2,3))},{'a':Value((2,3))}),[])

    def test_missing_active_rejected(self):
        with self.assertRaises(ValueError): check_state({}, {'a':Value((1,))})

    def test_shape_rejected(self):
        with self.assertRaises(ValueError): check_state({'a':Value((1,))},{'a':Value((2,))})

    def test_extra_rejected(self):
        with self.assertRaises(ValueError): check_state({'a':Value((1,))},{})

    def test_explicit_inactive_only(self):
        self.assertEqual(check_state({}, {'hair_gate':Value((1,))},('hair_gate',)),['hair_gate'])


if __name__ == '__main__': unittest.main()
