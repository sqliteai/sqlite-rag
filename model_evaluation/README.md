# Model Evaluation Python Script

A simple evaluation script for SQLite Rag using the MS MARCO dataset. Compares performance against the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) benchmarks.

## MS MARCO Dataset

**MS MARCO**: Microsoft Question-Answering dataset with real web queries and passages.

## Evaluation Metrics

- **Hit Rate (HR@k)**: Percentage of queries with relevant results in top-k
- **MRR**: Mean Reciprocal Rank - position-weighted relevance score
- **NDCG**: Normalized Discounted Cumulative Gain - ranking quality metric

## Usage

### 1. Setup Configuration

Create an example config file and then edit it with your model settings:

```bash
python ms_marco.py create-config
```

### 2. Process Dataset

```bash
python ms_marco.py process --config configs/my_config.json --limit-rows 100
```

Processes MS MARCO passages into the SQLite Rag database for evaluation.

### 3. Evaluate Performance

```bash
python ms_marco.py evaluate --config configs/my_config.json --limit-rows 100
```

Runs evaluation and saves results to `results/my_config_evaluation_results.txt`.

> **Note**: Without proper hardware, processing and evaluating the entire database may take a lot of time.
> Use `--limit-rows` to process and evaluate only the first n rows.

## Example Results

```
Metric               @1         @3         @5         @10
Hit Rate            0.650      0.780      0.820      0.850
MRR                 0.650      0.710      0.720      0.725
NDCG                0.650      0.715      0.735      0.750
```
