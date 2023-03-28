import pytest

from rl_agents.monte_carlo_tree_search.policies import MCTSPolicy


class TestMCTSPolicy:

    @pytest.mark.parametrize(
        "action_values, selection_count, expoloration, expected_output",
        [
            # Test for value of the output
            (
                {1: 1, 2: 1, 3: -1},
                {1: 49, 2: 49, 3: 2},
                3,
                {1: 1.6, 2: 1.6, 3: 9.0}
            ),
            # Test for unordered inputs
            (
                {2: 1, 3: -1, 1: 1},
                {3: 2, 1: 49, 2: 49},
                3,
                {1: 1.6, 2: 1.6, 3: 9.0}
            )
        ]
    )
    def test__compute_uct_values(self, action_values, selection_count, expoloration, expected_output):
        output = MCTSPolicy._compute_uct_values(action_values=action_values,
                                                selection_count=selection_count,
                                                exploration_cte=expoloration)
        assert output == expected_output

    # @pytest.mark.parametrize(
    #     "action_values, selection_count, expected_output",
    #     [
    #         # Test for value of the output
    #         (
    #             {1: 1, 2: 1, 3: -1},
    #             {1: 49, 2: 49, 3: 2},
    #             {1: 1.6, 2: 1.6, 3: 9.0}
    #         ),
    #         # Test for unordered inputs
    #         (
    #             {2: 1, 3: -1, 1: 1},
    #             {3: 2, 1: 49, 2: 49},
    #             {1: 1.6, 2: 1.6, 3: 9.0}
    #         )
    #     ]
    # )
    # def test__compute_adaptive_uct_values(self, action_values, selection_count, expected_output):
    #     output = MCTSPolicy._compute_adaptive_uct_values(action_values=action_values,
    #                                                      selection_count=selection_count)
    #     assert output == expected_output
