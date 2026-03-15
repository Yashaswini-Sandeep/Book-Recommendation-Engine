# Book Recommendation Engine

A recommendation engine built in R using real-world data from the [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) on Kaggle (400,000+ ratings).

---

## Techniques Implemented

| Method | Description |
|---|---|
| **User-Based Collaborative Filtering** | Finds users with similar reading tastes using Cosine Similarity and recommends books they loved |
| **Popularity-Based Fallback** | Recommends the highest-rated books overall — useful for new users (cold-start problem) |

---

## Repository Structure

```
Book-Recommendation-Engine/
│
├── SearchEngineRecommendation.R     # Main R script (fully modularised)
├── Books.csv                        # Book metadata (title, author, publisher)
├── Ratings.csv                      # User ratings (400k+ rows)
├── Users.csv                        # User demographics
├── Distribution of book ratings.png # Plot: rating distribution
├── Top Recommendations.png          # Plot: CF recommendations for top user
└── Top 10 popular books.png         # Plot: most popular books overall
```

---

## How It Works

```
Load Data -> Filter Active Users & Popular Items -> Build Sparse Matrix
    -> Compute Cosine Similarity -> Find Neighbours -> Weighted Score
        -> Return Top-N Recommendations
```

1. **Load** Ratings.csv and Books.csv
2. **Filter** users with 5 or more ratings and items with 5 or more ratings
3. **Build** a sparse user-item matrix (13,305 users x 14,513 books)
4. **Find** the most similar users using Cosine Similarity
5. **Score** unrated books using weighted average of neighbour ratings
6. **Return** Top 10 personalised recommendations with book titles

---

## Output Plots

### Rating Distribution
![Rating Distribution](Distribution%20of%20book%20ratings.png)

### Top 10 CF Recommendations
![Top Recommendations](Top%20Recommendations.png)

### Top 10 Most Popular Books
![Popular Books](Top%2010%20popular%20books.png)

---

## How to Run

### 1. Install R and RStudio
Download from https://posit.co/download/rstudio-desktop/

### 2. Clone the repository
```bash
git clone https://github.com/Yashaswini-Sandeep/Book-Recommendation-Engine.git
```

### 3. Open and run the script
Open SearchEngineRecommendation.R in RStudio, update the file paths if needed, then press Ctrl+A -> Ctrl+Enter.

### 4. Required packages (auto-installed)
- dplyr
- tidyr
- Matrix
- ggplot2

---

## Dataset

- **Source:** Kaggle - Book Recommendation Dataset
- **Ratings:** 433,671 explicit ratings (1-10 scale)
- **Books:** 271,360 unique books
- **Users:** 77,805 users

---

## Functions Overview

| Function | Purpose |
|---|---|
| `install_if_missing()` | Auto-installs required packages |
| `load_ratings()` | Loads and standardises Ratings.csv |
| `load_books()` | Loads Books.csv metadata |
| `build_rating_matrix()` | Builds sparse user-item matrix |
| `cosine_similarity()` | Computes cosine similarity between two users |
| `find_similar_users()` | Finds top-N most similar users |
| `recommend_cf()` | Generates CF recommendations |
| `recommend_popular()` | Generates popularity-based recommendations |
| `add_titles()` | Enriches results with book titles and authors |
| `plot_rating_dist()` | Plots rating distribution |
| `plot_cf_recs()` | Plots CF recommendations |
| `plot_popular_recs()` | Plots popular books |
| `run_engine()` | Master pipeline - runs everything |

---

## Author

**Yashaswini Sandeep**
Semester 1 - Computational Engineering
Programming Assignment - Recommendation Engine
