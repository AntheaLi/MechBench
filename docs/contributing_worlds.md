# Contributing Worlds

World contributors should not need to edit harness code.

1. Create a family if one does not already exist:

   ```bash
   python3 -m mechbench.cli new-family my_family
   ```

2. Create a world:

   ```bash
   python3 -m mechbench.cli new-world my_family/my_world
   ```

3. Fill in `world.yaml`:

   - `causal_label`
   - public `headline`
   - `budget`
   - private `interventions`
   - `experiment_table` or family-specific training hooks
   - `certificate`

4. Verify:

   ```bash
   python3 -m mechbench.cli verify my_family/my_world
   ```

The fixture family shows the minimum table-backed shape. Real v0 worlds should
come from the simulator compiler first, then the PyTorch retrieval family.

