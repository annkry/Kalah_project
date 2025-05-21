# Kalah AI project

This repo contains tools for testing and tuning MCTS/RAVE/GRAVE/PUCT/SHOT/SHUSS/NRPA/NMCS/MINIMAX/UNBOUNDED_MINIMAX strategies for the game of **Kalah** (Mancala).

---

## Main scripts

### `tuning.py`
Tunes hyperparameters for various AI methods (MCTS variants, NMCS).  
Generates plots of performance.

**Run with:**
```bash
python tuning.py
```

---

### `comparison_metrics.py`
Runs 1v1 matches between strategies, logs metrics, and shows win ratios.

**Run with:**
```bash
python comparison_metrics.py
```

---

### `tune_evaluate_weights.py`
Tunes weights for the heuristic evaluation function.

**Run with:**
```bash
python tune_evaluate_weights.py
```

---

## Other files

- `KalahGame.py`: handles board state, move logic, captures, and game rules.
- `KalahMCTS.py`: Monte Carlo Tree Search with RAVE, GRAVE, SHOT, SHUSS, and PUCT.
- `Kalahevaluate.py`: evaluation function (store diff, extra moves, captures).
- `KalahMinimax.py`: MINIMAX with alpha-beta pruning.
- `NRPA.py`: nested rollout policy adaptation.
- `NMCS.py`: nested Monte Carlo search.
- `UNMINIMAX.py`: unbounded minimax.

---

## Requirements
```bash
pip install -r requirements.txt
```

---

## Notes
- Plots are saved to `/plots`
- Tournament results saved to `tournament_results.csv`
- You can adjust game counts/iterations in the scripts for faster testing
