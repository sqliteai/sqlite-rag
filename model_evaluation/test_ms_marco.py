import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sqlite_rag import SQLiteRag


def test_ms_marco_processing(
    limit_rows=None, database_path="ms_marco_test.sqlite", rag_settings=None
):
    """Test processing MS MARCO dataset with SQLiteRag"""

    if rag_settings is None:
        rag_settings = {"chunk_size": 1000, "chunk_overlap": 0}

    # Load the MS MARCO test dataset
    parquet_path = Path("ms_marco_test.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset file {parquet_path} not found")

    df = pd.read_parquet(parquet_path)

    # Limit rows if specified
    if limit_rows:
        df = df.head(limit_rows)
        print(
            f"Loaded MS MARCO dataset with {len(df)} samples (limited from full dataset)"
        )
    else:
        print(f"Loaded MS MARCO dataset with {len(df)} samples")

    # Create SQLiteRag instance with provided settings
    rag = SQLiteRag.create(database_path, settings=rag_settings)

    # Process and add passages to the database
    total_passages_added = 0
    total_samples = len(df)

    print("Adding passages to sqlite_rag...")

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
            print(
                f"Processed {idx}/{total_samples} samples ({total_passages_added} passages)"
            )

    print(
        f"Finished! Added {total_passages_added} passages from {total_samples} queries"
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
    metrics = {k: {"hit_rate": 0, "reciprocal_ranks": []} for k in k_values}

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
        for k in k_values:
            top_k_results = search_results[:k]

            # Find relevant results in top-k
            relevant_found = False
            first_relevant_rank = None

            for rank, result in enumerate(top_k_results, 1):
                metadata = result.document.metadata
                if (
                    metadata
                    and metadata.get("query_id") == str(query_id)
                    and metadata.get("is_selected")
                ):

                    if not relevant_found:
                        relevant_found = True
                        first_relevant_rank = rank

            # Update hit rate
            if relevant_found:
                metrics[k]["hit_rate"] += 1

            # Update reciprocal rank (only for the specific k value)
            if first_relevant_rank and first_relevant_rank <= k:
                metrics[k]["reciprocal_ranks"].append(1.0 / first_relevant_rank)
            else:
                metrics[k]["reciprocal_ranks"].append(0.0)

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

    output(f"\n{'='*60}")
    output("INTERPRETATION:")
    output("- Hit Rate: % of queries where at least 1 relevant result appears in top-k")
    output("- MRR: Mean Reciprocal Rank, higher is better (max=1.0)")
    output("- Good performance: HR@5 > 0.7, MRR@5 > 0.5")
    output(f"{'='*60}")

    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write(f"Evaluation run on: {pd.Timestamp.now()}\n")
            f.write(f"Database: {database_path}\n")
            f.write(f"Limit rows: {limit_rows if limit_rows else 'All'}\n\n")
            f.write("\n".join(output_lines))

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
