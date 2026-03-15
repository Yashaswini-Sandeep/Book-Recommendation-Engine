# =============================================================================
# RECOMMENDATION ENGINE IN R
# Dataset: Book Recommendation Dataset (Kaggle)
# Techniques: Collaborative Filtering + Popularity-Based Fallback
# =============================================================================


# ── STEP 0: Install and load packages ────────────────────────────────────────

required_packages <- c("dplyr", "tidyr", "Matrix", "ggplot2")

install_if_missing <- function(pkgs) {
  missing_pkgs <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(missing_pkgs) > 0) {
    message("Installing: ", paste(missing_pkgs, collapse = ", "))
    install.packages(missing_pkgs, quiet = TRUE)
  }
  invisible(lapply(pkgs, library, character.only = TRUE))
}

install_if_missing(required_packages)


# ── STEP 1: Load Ratings ─────────────────────────────────────────────────────

load_ratings <- function(path) {
  message("Loading ratings...")
  df <- read.csv(path, stringsAsFactors = FALSE, sep = ",", quote = "\"")
  colnames(df) <- c("user_id", "item_id", "rating")
  df$rating <- as.numeric(df$rating)
  df <- df[!is.na(df$rating) & df$rating > 0, ]
  message(sprintf("Loaded %d ratings | %d users | %d items",
                  nrow(df), length(unique(df$user_id)), length(unique(df$item_id))))
  df
}


# ── STEP 2: Load Books ────────────────────────────────────────────────────────

load_books <- function(path) {
  message("Loading books...")
  df <- read.csv(path, stringsAsFactors = FALSE, sep = ",", quote = "\"")
  colnames(df)[1] <- "item_id"
  colnames(df)[2] <- "title"
  colnames(df)[3] <- "author"
  colnames(df)[4] <- "year"
  colnames(df)[5] <- "publisher"
  df <- df[, c("item_id", "title", "author", "year", "publisher")]
  message(sprintf("Loaded %d books", nrow(df)))
  df
}


# ── STEP 3: Build User-Item Matrix ────────────────────────────────────────────

build_rating_matrix <- function(ratings,
                                min_user_ratings = 5,
                                min_item_ratings = 5) {
  message("Building rating matrix...")

  user_counts <- table(ratings$user_id)
  item_counts <- table(ratings$item_id)

  active_users  <- names(user_counts[user_counts >= min_user_ratings])
  popular_items <- names(item_counts[item_counts >= min_item_ratings])

  filtered <- ratings[ratings$user_id %in% active_users &
                        ratings$item_id %in% popular_items, ]

  message(sprintf("Filtered: %d ratings | %d users | %d items",
                  nrow(filtered),
                  length(unique(filtered$user_id)),
                  length(unique(filtered$item_id))))

  user_fac <- factor(filtered$user_id)
  item_fac <- factor(filtered$item_id)

  mat <- sparseMatrix(
    i        = as.integer(user_fac),
    j        = as.integer(item_fac),
    x        = filtered$rating,
    dims     = c(nlevels(user_fac), nlevels(item_fac)),
    dimnames = list(levels(user_fac), levels(item_fac))
  )

  mat
}


# ── STEP 4: Cosine Similarity ─────────────────────────────────────────────────

cosine_similarity <- function(a, b) {
  both <- a != 0 & b != 0
  if (sum(both) < 2) return(NA_real_)
  a <- a[both]
  b <- b[both]
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}


# ── STEP 5: Find Similar Users (memory-efficient sparse version) ──────────────

find_similar_users <- function(mat, target_user, max_users = 5000) {
  if (!target_user %in% rownames(mat)) {
    stop("User not found: ", target_user)
  }

  target_vec <- as.numeric(mat[target_user, ])
  rated_by_target <- which(target_vec > 0)

  if (length(rated_by_target) == 0) {
    warning("Target user has no ratings in the filtered matrix.")
    return(numeric(0))
  }

  # Only consider users who rated at least one item the target user also rated
  other_users <- rownames(mat)[rownames(mat) != target_user]

  # Find users who share at least 1 item with target
  overlap <- apply(mat[other_users, rated_by_target, drop = FALSE], 1,
                   function(row) sum(row > 0))
  other_users <- names(overlap[overlap >= 1])

  if (length(other_users) == 0) {
    warning("No overlapping users found.")
    return(numeric(0))
  }

  if (length(other_users) > max_users) {
    other_users <- sample(other_users, max_users)
  }

  # Compute cosine similarity row by row
  sims <- sapply(other_users, function(uid) {
    row  <- as.numeric(mat[uid, ])
    both <- target_vec > 0 & row > 0
    if (sum(both) < 2) return(NA_real_)
    a <- target_vec[both]; b <- row[both]
    sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
  })

  sort(sims, decreasing = TRUE, na.last = TRUE)
}


# ── STEP 6: Collaborative Filtering Recommendations ──────────────────────────

recommend_cf <- function(mat, target_user, n_neighbors = 20, n_recs = 10) {
  message("Running collaborative filtering for user: ", target_user)

  sims      <- find_similar_users(mat, target_user)
  neighbors <- head(sims[!is.na(sims)], n_neighbors)

  if (length(neighbors) == 0) {
    warning("No similar users found.")
    return(NULL)
  }

  already_rated   <- colnames(mat)[mat[target_user, ] > 0]
  candidate_items <- setdiff(colnames(mat), already_rated)

  if (length(candidate_items) == 0) {
    message("User has already rated all items.")
    return(NULL)
  }

  neighbor_mat <- mat[names(neighbors), candidate_items, drop = FALSE]

  scores <- sapply(candidate_items, function(item) {
    item_ratings <- as.numeric(neighbor_mat[, item])
    rated        <- item_ratings > 0
    if (sum(rated) == 0) return(NA_real_)
    weighted.mean(item_ratings[rated], neighbors[rated])
  })

  result <- data.frame(
    item_id         = candidate_items,
    predicted_score = scores,
    stringsAsFactors = FALSE
  )
  result <- result[!is.na(result$predicted_score), ]
  result <- result[order(-result$predicted_score), ]
  result$rank <- seq_len(nrow(result))
  head(result, n_recs)
}


# ── STEP 7: Popularity-Based Recommendations (Cold Start) ────────────────────

recommend_popular <- function(ratings, n_recs = 10, min_ratings = 20) {
  message("Computing popularity-based recommendations...")

  agg <- aggregate(rating ~ item_id, data = ratings, FUN = function(x) {
    c(mean = mean(x), count = length(x))
  })

  df <- data.frame(
    item_id     = agg$item_id,
    mean_rating = agg$rating[, "mean"],
    n_ratings   = agg$rating[, "count"],
    stringsAsFactors = FALSE
  )

  df <- df[df$n_ratings >= min_ratings, ]
  df <- df[order(-df$mean_rating, -df$n_ratings), ]
  df$rank <- seq_len(nrow(df))
  head(df, n_recs)
}


# ── STEP 8: Add Book Titles to Results ───────────────────────────────────────

add_titles <- function(recs, books) {
  if (is.null(recs) || nrow(recs) == 0) return(recs)
  merged <- merge(recs, books[, c("item_id", "title", "author")],
                  by = "item_id", all.x = TRUE)
  merged[order(merged$rank), ]
}


# ── STEP 9: Plot Rating Distribution ─────────────────────────────────────────

plot_rating_dist <- function(ratings) {
  ggplot(ratings, aes(x = factor(rating))) +
    geom_bar(fill = "steelblue", color = "white") +
    labs(title = "Distribution of Book Ratings",
         x     = "Rating (1-10)",
         y     = "Number of Ratings") +
    theme_minimal(base_size = 13)
}


# ── STEP 10: Plot CF Recommendations ─────────────────────────────────────────

plot_cf_recs <- function(recs, target_user) {
  label_col <- if ("title" %in% names(recs)) "title" else "item_id"
  recs$label <- recs[[label_col]]
  recs$label <- factor(recs$label, levels = rev(recs$label))

  ggplot(recs, aes(x = label, y = predicted_score)) +
    geom_col(fill = "coral", color = "white") +
    coord_flip() +
    labs(title = paste("Top Recommendations for User", target_user),
         x     = NULL,
         y     = "Predicted Score") +
    theme_minimal(base_size = 12)
}


# ── STEP 11: Plot Popular Books ───────────────────────────────────────────────

plot_popular_recs <- function(recs) {
  label_col <- if ("title" %in% names(recs)) "title" else "item_id"
  recs$label <- recs[[label_col]]
  recs$label <- factor(recs$label, levels = rev(recs$label))

  ggplot(recs, aes(x = label, y = mean_rating)) +
    geom_col(fill = "mediumpurple", color = "white") +
    coord_flip() +
    labs(title = "Top 10 Most Popular Books",
         x     = NULL,
         y     = "Average Rating") +
    theme_minimal(base_size = 12)
}


# ── STEP 12: MAIN PIPELINE ────────────────────────────────────────────────────

run_engine <- function(ratings_path,
                       books_path,
                       target_user  = NULL,
                       n_recs       = 10) {

  # 1. Load data
  ratings <- load_ratings(ratings_path)
  books   <- load_books(books_path)

  # 2. Plot rating distribution
  print(plot_rating_dist(ratings))

  # 3. Build matrix
  mat <- build_rating_matrix(ratings,
                             min_user_ratings = 5,
                             min_item_ratings = 5)

  # 4. Pick user with most ratings if not given
  if (is.null(target_user)) {
    user_rating_counts <- rowSums(mat > 0)
    target_user <- names(which.max(user_rating_counts))
    message("Auto-selected most active user: ", target_user)
  }

  # 5. Collaborative filtering
  cf_recs <- recommend_cf(mat, target_user, n_recs = n_recs)
  cf_recs <- add_titles(cf_recs, books)

  message("\n====== COLLABORATIVE FILTERING RESULTS ======")
  if (!is.null(cf_recs) && nrow(cf_recs) > 0) {
    print(cf_recs)
    print(plot_cf_recs(cf_recs, target_user))
  } else {
    message("No CF recommendations generated for this user — showing popular instead.")
  }

  # 6. Popularity fallback
  pop_recs <- recommend_popular(ratings, n_recs = n_recs)
  pop_recs <- add_titles(pop_recs, books)

  message("\n====== POPULARITY-BASED RESULTS ======")
  print(pop_recs)
  print(plot_popular_recs(pop_recs))

  invisible(list(
    cf_recs  = cf_recs,
    pop_recs = pop_recs
  ))
}


# =============================================================================
# RUN THE ENGINE
# =============================================================================

results <- run_engine(
  ratings_path = "C:/Users/YASHASWINI SANDEEP/OneDrive/Desktop/Semester 1/2.Computational Engineering/3.Programming/Search_Engine/Ratings.csv",
  books_path   = "C:/Users/YASHASWINI SANDEEP/OneDrive/Desktop/Semester 1/2.Computational Engineering/3.Programming/Search_Engine/Books.csv",
  target_user  = NULL,
  n_recs       = 10
)
