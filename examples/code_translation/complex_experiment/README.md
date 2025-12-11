# Complex Experiment

5 branchpoints per method. 6 search strategies.

The point: with EnCompass, switching strategies is one config change.
With vanilla, you'd rewrite the whole agent.

```python
# switch strategy = change this line
config = StrategyConfig("beam_coarse", "beam", "greedy", width=3)
```

Strategies:
1. global_bon
2. local_bon_coarse
3. local_bon_fine
4. beam_coarse
5. bon_coarse_beam_fine
6. beam_coarse_beam_fine

```bash
python complex_experiment/run_all_strategies.py
```
