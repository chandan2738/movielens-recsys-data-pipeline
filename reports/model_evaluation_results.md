# Model Evaluation & Comparison Results

Primary cutoff: **K = 10** | Relevant item threshold: **rating >= 4.0**

## Side-by-Side Comparison (K = 10)

| Metric | Baseline Popularity | Hybrid Personalized SVD | Relative Improvement |
|---|---:|---:|---:|
| Precision@10 | 0.0650 | **0.1239** | +90.6% |
| Recall@10 | 0.0554 | **0.1360** | +145.5% |
| NDCG@10 | 0.0792 | **0.1675** | +111.5% |
| Catalog Coverage@10 | 0.0392 | **0.1718** | +338.3% |
| Intra-List Diversity@10 | 0.7127 | **0.5940** | -16.7% |
| RMSE (lower is better) | 1.0991 | **0.9703** | +11.7% |
| MAE (lower is better) | 0.8870 | **0.7690** | +13.3% |

## Sample Top-10 Hybrid Recommendations (User ID = 196)

| Rank | Item ID | Title | Genres | Score |
|---:|---:|---|---|---:|
| 1 | 168 | Monty Python and the Holy Grail (1974) | Comedy | 0.4606 |
| 2 | 216 | When Harry Met Sally... (1989) | Comedy|Romance | 0.4530 |
| 3 | 208 | Young Frankenstein (1974) | Comedy|Horror | 0.4154 |
| 4 | 154 | Monty Python's Life of Brian (1979) | Comedy | 0.4015 |
| 5 | 514 | Annie Hall (1977) | Comedy|Romance | 0.3959 |
| 6 | 83 | Much Ado About Nothing (1993) | Comedy|Romance | 0.3922 |
| 7 | 194 | Sting, The (1973) | Comedy|Crime | 0.3892 |
| 8 | 170 | Cinema Paradiso (1988) | Comedy|Drama|Romance | 0.3886 |
| 9 | 204 | Back to the Future (1985) | Comedy|Sci Fi | 0.3862 |
| 10 | 209 | This Is Spinal Tap (1984) | Comedy|Drama|Musical | 0.3845 |
