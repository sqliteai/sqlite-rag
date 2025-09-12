import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sqlite_rag import SQLiteRag


def calculate_dcg(relevance_scores):
    """Calculate Discounted Cumulative Gain"""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def calculate_ndcg(predicted_relevance, ideal_relevance):
    """Calculate Normalized Discounted Cumulative Gain"""
    if not predicted_relevance:
        return 0.0

    dcg = calculate_dcg(predicted_relevance)
    idcg = calculate_dcg(ideal_relevance)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def test_ms_marco_processing(
    limit_rows=None, database_path="ms_marco_test.sqlite", rag_settings=None
):
    """Test processing MS MARCO dataset with SQLiteRag"""

    start_time = time.time()

    if rag_settings is None:
        rag_settings = {"chunk_size": 1000, "chunk_overlap": 0}

    # Load the MS MARCO test dataset
    print("Loading MS MARCO dataset...")
    load_start = time.time()
    parquet_path = Path("ms_marco_test.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset file {parquet_path} not found")

    df = pd.read_parquet(parquet_path)
    load_time = time.time() - load_start

    # Limit rows if specified
    if limit_rows:
        df = df.head(limit_rows)
        print(
            f"Loaded MS MARCO dataset with {len(df)} samples (limited from full dataset) in {load_time:.2f}s"
        )
    else:
        print(f"Loaded MS MARCO dataset with {len(df)} samples in {load_time:.2f}s")

    if Path(database_path).exists():
        print(
            f"Warning: Database file {database_path} already exists and will be overwritten."
        )
        Path(database_path).unlink()

    # Create SQLiteRag instance with provided settings
    print("Initializing SQLiteRag...")
    init_start = time.time()
    rag = SQLiteRag.create(database_path, settings=rag_settings)
    init_time = time.time() - init_start
    print(f"SQLiteRag initialized in {init_time:.2f}s")

    # Process and add passages to the database
    total_passages_added = 0
    total_samples = len(df)
    processing_start = time.time()

    print(f"Adding passages to sqlite_rag... (processing {total_samples} queries)")

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        query_id = row["query_id"]
        query = row["query"]
        passages = row["passages"]

        # Extract passage texts and selected flags
        passage_texts = passages["passage_text"]
        is_selected = passages["is_selected"]
        urls = passages.get("url", [None] * len(passage_texts))

        # Add each passage as a separate document
        for i, (passage_text, selected, url) in enumerate(
            zip(passage_texts, is_selected, urls)
        ):
            # Create metadata with relevance information
            metadata = {
                "query_id": str(query_id),
                "query": query,
                "is_selected": bool(selected),
                "passage_index": i,
                "url": url if url else None,
            }

            # Create URI for the passage
            uri = f"ms_marco_query_{query_id}_passage_{i}"

            # Add passage to the database
            rag.add_text(text=passage_text, uri=uri, metadata=metadata)

            total_passages_added += 1

        # Progress update every 100 samples
        if idx % 100 == 0:
            elapsed = time.time() - processing_start
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total_samples - idx) / rate if rate > 0 else 0
            print(
                f"Progress: {idx}/{total_samples} queries ({idx/total_samples*100:.1f}%) | "
                f"{total_passages_added} passages | {rate:.1f} queries/s | "
                f"ETA: {eta/60:.1f}m"
            )

    processing_time = time.time() - processing_start
    print(
        f"Processing completed! Added {total_passages_added} passages from {total_samples} queries in {processing_time:.2f}s"
    )

    # Verify data was added correctly
    documents = rag.list_documents()
    print(f"Total documents in database: {len(documents)}")

    # Show sample of added documents
    if documents:
        sample_doc = documents[0]
        print("\nSample document:")
        print(f"URI: {sample_doc.uri}")
        print(f"Content (first 100 chars): {sample_doc.content[:100]}...")
        print(f"Metadata: {sample_doc.metadata}")

    # Store query information for future search evaluation
    queries = df[["query_id", "query"]].drop_duplicates()
    queries_file = Path("ms_marco_queries.json")
    queries_dict = queries.to_dict("records")

    with open(queries_file, "w") as f:
        json.dump(queries_dict, f, indent=2)

    print(f"\nSaved {len(queries_dict)} unique queries to {queries_file}")
    print("Ready for search evaluation!")

    # Show settings for verification
    settings_info = rag.get_settings()
    print("\nCurrent settings:")
    print(f"  chunk_size: {settings_info['chunk_size']}")
    print(f"  chunk_overlap: {settings_info['chunk_overlap']}")
    print(f"  weight_fts: {settings_info.get('weight_fts', 1.0)}")
    print(f"  weight_vec: {settings_info.get('weight_vec', 1.0)}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("TIMING SUMMARY:")
    print(f"  Dataset loading: {load_time:.2f}s")
    print(f"  SQLiteRag init:  {init_time:.2f}s")
    print(f"  Processing:      {processing_time:.2f}s")
    print(f"  Total time:      {total_time:.2f}s")
    print(f"  Average rate:    {total_passages_added/processing_time:.1f} passages/s")
    print(f"{'='*60}")

    rag.close()
    return total_passages_added, len(queries_dict)


def evaluate_search_quality(
    limit_rows=None, database_path="ms_marco_test.sqlite", output_file=None
):
    """Evaluate search quality using proper metrics"""

    # Setup output capture
    output_lines = []

    def output(text):
        """Print to console and capture for file output"""
        print(text)
        output_lines.append(text)

    # Load the dataset directly
    parquet_path = Path("ms_marco_test.parquet")
    if not parquet_path.exists():
        output("Dataset file not found. Run 'process' action first.")
        return

    df = pd.read_parquet(parquet_path)

    # Apply same row limit as processing
    if limit_rows:
        df = df.head(limit_rows)
        output(f"Evaluating on {len(df)} queries (limited from full dataset)")
    else:
        output(f"Evaluating on {len(df)} queries")

    # Create RAG instance
    rag = SQLiteRag.create(database_path)

    # Metrics for different top-k values
    k_values = [1, 3, 5, 10]
    metrics = {
        k: {"hit_rate": 0, "reciprocal_ranks": [], "ndcg_scores": []} for k in k_values
    }

    # Track queries with no matches at HR@1 for detailed output
    failed_hr1_queries = []

    total_queries = 0
    queries_with_relevant = 0

    output("\nRunning evaluation...")

    for idx, (_, row) in enumerate(df.iterrows()):
        query_id = row["query_id"]
        query_text = row["query"]
        passages = row["passages"]

        # Get ground truth (which passages are marked as selected)
        selected_indices = set(np.where(passages["is_selected"] == 1)[0])

        if not selected_indices:
            continue  # Skip queries with no relevant passages

        queries_with_relevant += 1
        total_queries += 1

        # Perform search
        search_results = rag.search(query_text, top_k=10)

        # Check results for each k value
        hr1_found = False  # Track if any relevant result found in top-1

        for k in k_values:
            top_k_results = search_results[:k]

            # Find relevant results in top-k
            relevant_found = False
            first_relevant_rank = None
            predicted_relevance = []  # For NDCG calculation

            for rank, result in enumerate(top_k_results, 1):
                metadata = result.document.metadata
                is_relevant = (
                    metadata
                    and metadata.get("query_id") == str(query_id)
                    and metadata.get("is_selected")
                )

                # Add relevance score (1 for relevant, 0 for non-relevant)
                predicted_relevance.append(1.0 if is_relevant else 0.0)

                if is_relevant:
                    if not relevant_found:
                        relevant_found = True
                        first_relevant_rank = rank

                        # Track HR@1 success
                        if k == 1:
                            hr1_found = True

            # Calculate NDCG@k
            # Ideal relevance: all relevant docs at the top
            num_relevant = len(selected_indices)
            ideal_relevance = [1.0] * min(num_relevant, k) + [0.0] * max(
                0, k - num_relevant
            )
            ndcg_score = calculate_ndcg(predicted_relevance, ideal_relevance)
            metrics[k]["ndcg_scores"].append(ndcg_score)

            # Update hit rate
            if relevant_found:
                metrics[k]["hit_rate"] += 1

            # Update reciprocal rank (only for the specific k value)
            if first_relevant_rank and first_relevant_rank <= k:
                metrics[k]["reciprocal_ranks"].append(1.0 / first_relevant_rank)
            else:
                metrics[k]["reciprocal_ranks"].append(0.0)

        # Track queries that failed HR@1
        if not hr1_found:
            failed_hr1_queries.append({"query_id": query_id, "query": query_text})

        # Progress update
        if (idx + 1) % 50 == 0:
            print(
                f"Processed {idx + 1}/{len(df)} queries..."
            )  # Only to console, not to file

    rag.close()

    # Calculate and display final metrics
    output(f"\n{'='*60}")
    output("SEARCH QUALITY EVALUATION RESULTS")
    output(f"{'='*60}")
    output(f"Total queries evaluated: {queries_with_relevant}")
    output(f"Queries with relevant passages: {queries_with_relevant}")

    output(f"\n{'Metric':<20} {'@1':<10} {'@3':<10} {'@5':<10} {'@10':<10}")
    output("-" * 60)

    # Hit Rate (HR@k)
    hit_rates = []
    for k in k_values:
        if queries_with_relevant > 0:
            hr = metrics[k]["hit_rate"] / queries_with_relevant
            hit_rates.append(f"{hr:.3f}")
        else:
            hit_rates.append("0.000")

    output(
        f"{'Hit Rate':<20} {hit_rates[0]:<10} {hit_rates[1]:<10} {hit_rates[2]:<10} {hit_rates[3]:<10}"
    )

    # Mean Reciprocal Rank (MRR@k)
    mrr_values = []
    for k in k_values:
        if metrics[k]["reciprocal_ranks"]:
            mrr = np.mean(metrics[k]["reciprocal_ranks"])
            mrr_values.append(f"{mrr:.3f}")
        else:
            mrr_values.append("0.000")

    output(
        f"{'MRR':<20} {mrr_values[0]:<10} {mrr_values[1]:<10} {mrr_values[2]:<10} {mrr_values[3]:<10}"
    )

    # NDCG@k
    ndcg_values = []
    for k in k_values:
        if metrics[k]["ndcg_scores"]:
            ndcg = np.mean(metrics[k]["ndcg_scores"])
            ndcg_values.append(f"{ndcg:.3f}")
        else:
            ndcg_values.append("0.000")

    output(
        f"{'NDCG':<20} {ndcg_values[0]:<10} {ndcg_values[1]:<10} {ndcg_values[2]:<10} {ndcg_values[3]:<10}"
    )

    output(f"\n{'='*60}")
    output("INTERPRETATION:")
    output("- Hit Rate: % of queries where at least 1 relevant result appears in top-k")
    output("- MRR: Mean Reciprocal Rank, higher is better (max=1.0)")
    output(
        "- NDCG: Normalized Discounted Cumulative Gain, considers relevance and position (max=1.0)"
    )
    output("- Good performance: HR@5 > 0.7, MRR@5 > 0.5, NDCG@5 > 0.6")
    output(f"{'='*60}")

    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write(f"Evaluation run on: {pd.Timestamp.now()}\n")
            f.write(f"Database: {database_path}\n")
            f.write(f"Limit rows: {limit_rows if limit_rows else 'All'}\n\n")
            f.write("\n".join(output_lines))

            # Add list of queries that failed HR@1
            if failed_hr1_queries:
                f.write(f"\n\n{'='*60}\n")
                f.write(
                    f"QUERIES WITH NO MATCHES AT HR@1 ({len(failed_hr1_queries)} queries):\n"
                )
                f.write(f"{'='*60}\n\n")
                for i, query_info in enumerate(failed_hr1_queries, 1):
                    f.write(f"{i}. Query ID: {query_info['query_id']}\n")
                    f.write(f"   Query: {query_info['query']}\n\n")

        print(f"\nResults saved to: {output_file}")


def search_evaluation_example():
    """Legacy function - use evaluate_search_quality() instead"""
    return evaluate_search_quality()


def load_config(config_path):
    """Load configuration from JSON file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")

    with open(config_file, "r") as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ["database_path"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from config file")

    return config


def create_example_config():
    """Create an example configuration file"""
    example_config = {
        "database_path": "ms_marco_test.sqlite",
        "rag_settings": {
            "chunk_size": 1000,
            "chunk_overlap": 0,
            "weight_fts": 1.0,
            "weight_vec": 1.0,
            "model_path_or_name": "./models/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
            "model_options": "",
            "model_context_options": "generate_embedding=1,normalize_embedding=1,pooling_type=mean,embedding_type=INT8",
            "vector_type": "INT8",
            "embedding_dim": 1024,
            "other_vector_options": "distance=cosine",
        },
    }

    config_file = Path("ms_marco_config.json")
    with open(config_file, "w") as f:
        json.dump(example_config, f, indent=2)

    print(f"Created example configuration file: {config_file}")
    return config_file


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Process MS MARCO dataset with SQLiteRag or evaluate search quality"
    )
    parser.add_argument(
        "action",
        choices=["process", "evaluate"],
        help="Action to perform: 'process' to add passages to database, 'evaluate' to test search quality",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Limit the number of rows (queries) to process from the dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON configuration file with RAG settings and database path",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file {args.config} not found. Creating example config...")
        create_example_config()
        print("Please edit ms_marco_config.json with your settings and try again.")
        return
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Extract settings
    database_path = config["database_path"]
    rag_settings = config.get("rag_settings", {})

    if args.action == "process":
        limit_msg = f" (limited to {args.limit_rows} rows)" if args.limit_rows else ""
        print(f"Processing MS MARCO dataset{limit_msg}")
        print(f"Database: {database_path}")
        print(f"Settings: {rag_settings}")

        passages_added, queries_saved = test_ms_marco_processing(
            limit_rows=args.limit_rows,
            database_path=database_path,
            rag_settings=rag_settings,
        )

        print("\nSummary:")
        print(f"- Added {passages_added} passages to the database")
        print(f"- Saved {queries_saved} unique queries for evaluation")
        print(f"- Database file: {database_path}")
        print("- Queries file: ms_marco_queries.json")

        print("\nNext steps:")
        print(f"python test_ms_marco.py evaluate --config {args.config}")

    elif args.action == "evaluate":
        print("Evaluating search quality...")
        print(f"Database: {database_path}")

        # Generate output filename based on config file
        config_name = Path(args.config).stem
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / f"{config_name}_evaluation_results.txt"

        evaluate_search_quality(
            limit_rows=args.limit_rows,
            database_path=database_path,
            output_file=output_file,
        )


if __name__ == "__main__":
    main()
