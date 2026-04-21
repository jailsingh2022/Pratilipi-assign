# Recommendation System Write-Up

## Problem Framing
The task requires predicting what a user should read next based on their reading history. A critical factor in this dataset is that chapters are sequential. Recommending the 15th chapter of a book to a user who has never read the book is a poor user experience. Therefore, I divided the recommendation problem into two distinct use-cases for a hybrid model:
1. **Continue Reading (Exploitation)**: If a user has interacted with a book but hasn't reached the end, the most logical recommendation is the chronologically next chapter in that specific book. 
2. **Start a New Book (Exploration)**: If a user needs a new book, we should recommend the *first chapter* of a book they haven't started reading yet.

## Methodology

### 1. Continue Reading Recommender
* **Approach**: Rule-based sequences.
* **Mechanism**: The system groups user interactions by `book_id` and calculates the maximum `chapter_sequence_no` the user has read. If the book has subsequent chapters (by looking at the overall maximum sequence number for that book in the chapters dataset), it recommends max sequence + 1. 

### 2. New Book Recommender
* **Approach**: Content-Based Filtering & Popularity Fallback.
* **Mechanism**: 
    - **Item Profiles**: We represent each book by aggregating all the `tags` of its combined chapters and treating the `author_id` as part of the meta-document. A TF-IDF vectorizer builds the feature matrix for all 50,000+ books.
    - **User Profiles**: For a given user, we take the books they've interacted with and calculate the mean of the TF-IDF vectors of those books. This forms a "User Profile" spanning the tag representation space.
    - **Scoring**: A cosine similarity computation is run between the User Profile and all unseen Book Profiles to return the top K recommendations.
    - **Cold-Start Handling**: If the user has absolutely no history (or the interacted books are inexplicably missing from the metadata), the model falls back to a global popularity recommender that suggests the most frequently read books across the entire dataset.

## Evaluation
Due to the absence of exact timestamps down to the second/minute in the `interactions.csv` dataset, building a rigorous chronological timeline across different books is difficult. Thus, I structured the evaluation using an **Offline Leave-One-Out (Hit Rate @ K)** strategy on the "New Book" prediction component.
- We filter to users who have read multiple books.
- We hide one entire book randomly from their history.
- The model outputs Top K new book recommendations on the remaining history.
- If the held-out book is in the list, it's counted as a 'hit'.

**Results:** In local evaluation on a subset of users, Content-based filtering achieved an HR@10 of `0.0020` on the first iteration. This is a baseline indicating sheer data sparsity (1 million interactions, 150k users equals an average of ~8 rows per user; realistically only 1-3 unique books per user tail).  

## Tradeoffs and Architectural Decisions
- **Computational Constraints & Memory Limits**: With highly sparse interactions (150k unique users, 50k books), generating a dense `150k x 50k` similarity matrix consumes gigabytes of RAM. Hence, rather than memory-based collaborative filtering dense matrix operations, the model uses sparse vectors dynamically on demand per user ensuring maximum scalability within constraints.
- **Why TF-IDF instead of Matrix Factorization?**: With an extreme sparsity level (0.013%), collaborative filtering architectures like ALS/SVD often suffer from severe cold-start issues. Content-based TF-IDF acts robustly out-of-the-box leveraging specific metadata (tags, author bias).

## What I'd Improve Given More Time
1. **Implicit Feedback Collaborative Filtering (ALS)**: I would train an Alternating Least Squares (ALS) model on the user-item interaction matrix, leveraging algorithms like `implicit` library to capture unobserved interactions.
2. **Deep Sequential Models**: With more time, training a transformer based sequence model (e.g. BERT4Rec or SASRec) utilizing sequence and padding tokens would naturally learn within-book and cross-book transition logic seamlessly overriding hard-coded logic.
3. **Model Ensembling**: Compute blended recommendations blending Content-Based representations with Popularity scores weighted by a time-decay algorithm if exact timestamps were captured in pipelines.
