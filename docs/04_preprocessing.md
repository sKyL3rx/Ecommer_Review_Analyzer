# Preprocessing and Feature Engineering

The preprocessing flow is split into three  parts:

```text
raw reviews + product metadata
    -> Phase 1 review cleaning and sentiment labels
    -> Phase 2 product-sentiment review groups
    -> synthetic summary examples and TRL/SFT splits
```

## 1. Phase 1: review-level sentiment dataset

**Script:** `src/data/preprocess_phase1.py`

Phase 1 creates the cleaned review-level dataset used for sentiment training and later summary grouping.

### Inputs

```text
SAMPLE_N = 500000 ( Used 500000 samples for this project)
data/raw/appliances_reviews_sample_{SAMPLE_N}.csv
data/raw/appliances_meta_matched.csv
```

The review file is expected to contain these columns:

```text
product_id
rating
review_title
review_text
timestamp
helpful_vote
verified_purchase
```

### What the script does

The script performs the following steps:

1. Validates the required review columns.
2. Drops rows missing `product_id`, `rating`, or `review_text`.
3. Cleans `review_title` and `review_text`.
4. Converts `rating`, `timestamp`, `helpful_vote`, and `verified_purchase` to usable numeric/boolean-style fields.
5. Builds a single lowercase text field named `text` from title and body.
6. Computes `text_len` as the token count of the cleaned text.
7. Filters out very short reviews using `min_review_len`.
8. Keep only English reviews using `py3langid` and a confidence threshold.
9. Removes duplicate reviews by `product_id`, `rating`, `review_text`, and `timestamp`.
10. Maps star ratings into `positive`, `neutral`, and `negative` sentiment labels.
11. Joins selected product metadata columns from the metadata file.
12. Creates stratified train/validation/test splits by `sentiment_label`.

### Text cleaning

- decodes HTML entities,
- removes HTML tags,
- removes punctuation except `?` and `!`,
- adds spacing around `?` and `!`,
- normalizes whitespace,
- lowercases the final model text.

### Sentiment labels

The sentiment label is derived from the review rating:

```text
rating >= 4  -> positive
rating == 3  -> neutral
rating < 3   -> negative
```


### Main output columns

| Column | Meaning |
|---|---|
| `review_id` | Hash-based review identifier. |
| `product_id` | Product identifier. |
| `rating` | Numeric review rating. |
| `review_title` | Cleaned review title. |
| `review_text` | Cleaned review body. |
| `text` | Lowercased title/body text used for modeling. |
| `sentiment_label` | Sentiment class. |
| `timestamp` | Review timestamp. |
| `helpful_vote` | Numeric helpful vote count. |
| `verified_purchase` | Integer purchase signal. |
| `text_len` | Token count of `text`. |
| product metadata columns | Joined when available, for example `product_title`, `store`, `main_category`, `price`. |

### Outputs

```text
data/interim/reviews_clean.csv

data/processed/phase1/train_val.csv
data/processed/phase1/train.csv
data/processed/phase1/val.csv
data/processed/phase1/test.csv

data/processed/phase1/folds/fold_*/train.csv
data/processed/phase1/folds/fold_*/val.csv
```

### Example command

```bash
python -m src.data.preprocess_phase1 \
  --input_reviews data/raw/appliances_reviews_sample_500000k.csv \
  --input_meta data/raw/appliances_meta_matched.csv \
  --interim_dir data/interim \
  --processed_dir data/processed/phase1 \
  --english_only \
  --lang_conf_threshold 0.90 \
  --n_splits 5 \
  --save_kfold \
  --test_size 0.15 \
  --val_size 0.15 \
  --random_state 42 \
  --min_review_len 3
```

---


## 2. Phase 2: product-sentiment groups for summarization

**Script:** `src/data/preprocess_phase2.py`

Phase 2 converts cleaned review-level data into product-level review groups. Each product can have positive, neutral, and negative groups, with a small set of representative reviews selected from each group.

This dataset is used to build summary prompts and SFT training examples.

### Input

```text
data/interim/reviews_clean.csv
```

The script validates that the input contains:

```text
review_id
product_id
rating
text
sentiment_label
timestamp
helpful_vote
verified_purchase
text_len
```

### Review quality score

The script computes a `review_quality_score` to rank candidate reviews inside each `(product_id, sentiment_label)` group.

The score combines:

- helpful votes, after `log1p` scaling and min-max normalization,
- verified-purchase signal,
- review length score.

```text
review_quality_score =
  (weight_helpful  * helpful_vote_norm
 + weight_verified * verified_purchase
 + weight_length   * length_score)
 / total_weight
```

By default, the weights are:

```text
helpful_vote:      0.40
verified_purchase: 0.30
length_score:      0.30
```


### Representative review selection

For each product-sentiment group, the script:

1. skips groups smaller than `min_group_size`,
2. sorts reviews by quality score, helpful votes, verified-purchase flag, and length,
3. keeps up to `max_reviews_per_group` reviews,
4. optionally uses sentence embeddings to avoid selecting near-duplicate reviews,
5. fills back with the next best reviews if the diversity filter becomes too strict.

When embeddings are enabled, the script uses `sentence-transformers/all-MiniLM-L6-v2` by default and filters candidates using cosine similarity, using this to avoid similar reviews.

### Outputs

```text
data/processed/phase2/product_sentiment_groups_ranked.jsonl
data/processed/phase2/product_sentiment_group_stats_ranked.csv
```

The JSONL file keeps the full selected review lists. Each product record can include:

```text
product_id
product_title
store
main_category
review_count
avg_rating_from_reviews
positive_count / neutral_count / negative_count
positive_ratio / neutral_ratio / negative_ratio
positive_reviews / neutral_reviews / negative_reviews
positive_review_items / neutral_review_items / negative_review_items
```

The CSV file is a lighter stats file. It drops large list fields and keeps scalar group-level metadata and counts.

### Example command

```bash
python -m src.data.preprocess_phase2 \
  --input_path data/interim/reviews_clean.csv \
  --output_jsonl data/processed/phase2/product_sentiment_groups_ranked.jsonl \
  --output_csv data/processed/phase2/product_sentiment_group_stats_ranked.csv \
  --min_reviews_per_group 3 \
  --max_reviews_per_group 6 \
  --min_group_size 3 \
  --similarity_threshold 0.82 \
  --ideal_min_len 20 \
  --ideal_max_len 100 \
  --hard_max_len 250 \
  --use_embeddings \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

---

## 3. Synthetic summary dataset generation

**Script:** `scripts/generate_synthetic_gold_summary_via_api.py`

This script creates instruction-style summary examples from the Phase 2 product-sentiment groups. I used the model "gemini-2.5-flash-lite" to create dataset for Phase 2.
### Input

```text
data/processed/phase2/product_sentiment_groups_ranked.jsonl
```

### What the script does

For each product and sentiment group, the script:

1. checks whether enough sentiment groups are present using `min_sentiments_present`,
2. selects up to `top_k` representative reviews,
3. deduplicates selected review text while preserving order,
4. builds a prompt with product title, category, sentiment, and numbered reviews,
5. sends the prompt to either OpenAI or Gemini,
6. writes the generated summary as `gold_summary`,

The prompt explicitly asks the generator to:

- use only information supported by the reviews,
- avoid invented product details,
- focus on recurring themes,
- keep the answer concise.

### Output row fields

Each successful JSONL row contains fields such as:

```text
product_id
product_title
main_category
sentiment
source_reviews
source_review_count
original_sentiment_count
prompt_hash
prompt
gold_summary
generator_provider
generator_model
```

### Outputs

```text
data/processed/phase2/phase2_sft_summary_dataset.jsonl
data/processed/phase2/phase2_sft_summary_errors.jsonl
```

### Example command

```bash
python -m scripts.generate_synthetic_gold_summary_via_api \
  --input_path data/processed/phase2/product_sentiment_groups_ranked.jsonl \
  --output_path data/processed/phase2/phase2_sft_summary_dataset.jsonl \
  --errors_path data/processed/phase2/phase2_sft_summary_errors.jsonl \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --top_k 5 \
  --min_sentiments_present 2 \
  --sleep_seconds 0.3 \
  --resume \
  --max_workers 4
```

---

## 4. TRL/SFT split builder

**Script:** `src/data/preprocess_phase2_sft_trl.py`

This script converts the synthetic summary dataset into chat-format JSONL files ( matching with TRL libray format) for SFT.

### Input

```text
data/processed/phase2/phase2_sft_summary_dataset.jsonl
```

The script uses these columns:

```text
product_id
sentiment
prompt
gold_summary
```

### Split strategy

The script uses `GroupShuffleSplit` with `product_id` as the group. This prevents the same product from appearing across train, validation, and test splits.

The split is:

```text
80% train
10% validation
10% test
```

### Output format

Each row is exported as a `messages` object:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

### Outputs

```text
data/processed/phase2/trl_data/train.jsonl
data/processed/phase2/trl_data/val.jsonl
data/processed/phase2/trl_data/test.jsonl
```


## 5. Serving table preparation

### Product catalog table

**Script:** `src/serving/build_catalog.py`

This script combines product metadata with aggregated review statistics.

It creates fields such as:

```text
product_id
product_title
store
main_category
price
average_rating
rating_number
review_count
avg_review_rating
verified_review_count
total_helpful_votes
latest_review_ts
image_url
categories_list
features_text
description_text
search_text
```

Output:

```text
data/serving/appliances_demo_catalog.parquet
```

### Review serving table

**Script:** `src/serving/build_reviews_index.py`

This script prepares review rows for online product insight generation.

It creates fields such as:

```text
review_id
product_id
review_title
review_text
review_char_len
timestamp
rating
review_datetime
helpful_vote
verified_purchase
has_review_image
```

Output:

```text
data/serving/appliances_demo_reviews.parquet
```

---