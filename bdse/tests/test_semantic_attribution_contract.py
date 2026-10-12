"""Regression tests for optional ICER attribution diagnostics."""
from __future__ import annotations

import numpy as np
import unittest

from bdse.planner.semantic_attribution import (
    family_feature_matrix,
    type_feature_matrix,
)


class SemanticAttributionContractTests(unittest.TestCase):
    def test_metadata_without_attribution_rows_is_a_valid_unavailable_view(self):
        valid = np.asarray([True, True, True])
        family, family_names = family_feature_matrix(
            None, [1, 2, 3, 4, 5, 2, 2, 3], valid, 0, 1
        )
        atom_type, type_names = type_feature_matrix(
            None, ["occupancy"] * 8, valid, 0, 1
        )

        self.assertEqual(family.shape, (3, len(family_names)))
        self.assertEqual(atom_type.shape, (3, len(type_names)))
        self.assertEqual(np.count_nonzero(family), 0)
        self.assertEqual(np.count_nonzero(atom_type), 0)

    def test_nonempty_attribution_still_requires_exact_metadata_alignment(self):
        contrib = np.zeros((2, 3), dtype=np.float64)
        valid = np.asarray([True, True, True])

        with self.assertRaisesRegex(ValueError, "one family id per selected atom"):
            family_feature_matrix(contrib, [1], valid, 0, 1)
        with self.assertRaisesRegex(ValueError, "one atom type per selected attribution row"):
            type_feature_matrix(contrib, ["occupancy"], valid, 0, 1)


if __name__ == "__main__":
    unittest.main()
