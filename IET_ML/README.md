# ITA0614 Air Quality ML Project

## Files
- `air_quality.csv` — synthetic/demo dataset matching the assignment columns.
- `air_quality_ml.py` — complete runnable program covering preprocessing, EDA, Candidate-Elimination demo, Decision Tree, Gaussian Naive Bayes, first-principles KNN, first-principles LWR, first-principles MLP Back Propagation, GA feature selection, evaluation and warning logic.
- `requirements.txt` — Python packages.

## Important
The assignment document requires a real air-quality dataset and its exact source/license to be documented. The included CSV is only a demo dataset so the code can run immediately. Replace it with the real dataset before final submission.

## Run in VS Code
1. Install Python 3.10+.
2. Open this folder in VS Code.
3. Open Terminal.
4. Run:
   `pip install -r requirements.txt`
5. Run:
   `python air_quality_ml.py`

## Outputs
The program creates:
- `results/class_summary.csv`
- `results/lwr_results.csv`
- `results/model_comparison.csv`
- `results/figures/` with the required plots and confusion matrices.
