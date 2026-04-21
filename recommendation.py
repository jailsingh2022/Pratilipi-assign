import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class SimpleTFIDF:
    def __init__(self):
        self.doc_freqs = defaultdict(int)
        self.doc_count = 0
        self.idf = {}
        
    def fit_transform(self, docs):
        """Calculates TF-IDF from scratch given a list of lists of tokens"""
        self.doc_count = len(docs)
        
        # Calculate Document Frequency (DF)
        for doc in docs:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] += 1
                
        # Calculate Inverse Document Frequency (IDF)
        for term, df in self.doc_freqs.items():
            # Adding 1 to smooth
            self.idf[term] = math.log((1 + self.doc_count) / (1 + df)) + 1
            
        # Create normalized TF-IDF sparse vectors
        tfidf_vectors = []
        for doc in docs:
            term_counts = Counter(doc)
            vec = {}
            norm = 0.0
            for term, count in term_counts.items():
                tfidf_weight = count * self.idf[term]
                vec[term] = tfidf_weight
                norm += tfidf_weight ** 2
            
            norm = math.sqrt(norm)
            if norm > 0:
                for term in vec:
                    vec[term] /= norm
            
            tfidf_vectors.append(vec)
            
        return tfidf_vectors

def average_sparse_vectors(vec_list):
    avg_vec = defaultdict(float)
    n = len(vec_list)
    if n == 0: return dict(avg_vec)
    for vec in vec_list:
        for term, weight in vec.items():
            avg_vec[term] += weight / n
    return dict(avg_vec)

def cosine_similarity_sparse(vec1, vec2):
    # Both vectors are assumed to be dictionaries mapping term -> weight
    # and they should be L2 normalized.
    score = 0.0
    # Iterate over the smaller vector to optimize
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1
        
    for term, weight in vec1.items():
        if term in vec2:
            score += weight * vec2[term]
    return score

class DataProcessor:
    def __init__(self, chapters_file='chapters.csv', interactions_file='interactions.csv'):
        self.chapters_file = chapters_file
        self.interactions_file = interactions_file
        self.chapters = None
        self.interactions = None
        self.book_metadata = None

    def load_data(self):
        print(f"Loading {self.chapters_file}...")
        self.chapters = pd.read_csv(self.chapters_file)
        
        print(f"Loading {self.interactions_file}...")
        self.interactions = pd.read_csv(self.interactions_file)
        
        print("Processing book metadata...")
        self.chapters['tags'] = self.chapters['tags'].fillna('')
        
        # Aggregate chapters to books
        self.book_metadata = self.chapters.groupby('book_id').agg({
            'author_id': 'first', 
            'tags': lambda x: '|'.join(set('|'.join(x).split('|'))),
            'chapter_sequence_no': 'max'
        }).reset_index()
        self.book_metadata.rename(columns={'chapter_sequence_no': 'total_chapters'}, inplace=True)
        # Clean tags
        self.book_metadata['tags'] = self.book_metadata['tags'].str.replace('\|', ' ', regex=True)

class ContinueReadingRecommender:
    def __init__(self, chapters_df):
        self.book_max_chapters = chapters_df.groupby('book_id')['chapter_sequence_no'].max().to_dict()
        
    def recommend(self, user_interactions_df):
        """
        Recommends the next chapter in books the user has started but not finished.
        """
        user_progress = user_interactions_df.groupby('book_id')['chapter_sequence_no'].max()
        
        recommendations = {}
        for book_id, max_read_seq in user_progress.items():
            max_available = self.book_max_chapters.get(book_id, max_read_seq)
            if max_read_seq < max_available:
                recommendations[book_id] = max_read_seq + 1
                
        return recommendations

class NewBookRecommender:
    def __init__(self, book_metadata_df):
        self.book_metadata = book_metadata_df
        self.books = book_metadata_df['book_id'].values
        
        print("Generating TF-IDF matrices from scratch...")
        text_features = book_metadata_df['tags'] + " author_" + book_metadata_df['author_id'].astype(str)
        # Tokenize by whitespace
        docs = [str(text).split() for text in text_features]
        
        self.tfidf = SimpleTFIDF()
        self.book_vectors = self.tfidf.fit_transform(docs)
        self.popular_books = []
        
    def prepare_popularity_baseline(self, interactions_df):
        pop = interactions_df['book_id'].value_counts()
        self.popular_books = pop.index.tolist()

    def recommend(self, user_interacted_books, top_k=5):
        """
        Recommends top_k new books for a user based on their interacted books (Content-Based from scratch)
        """
        if len(user_interacted_books) == 0:
            return self.popular_books[:top_k]
            
        interacted_indices_set = set(self.book_metadata.index[self.book_metadata['book_id'].isin(user_interacted_books)])
        interacted_indices = list(interacted_indices_set)
        
        if not interacted_indices:
            return self.popular_books[:top_k]
            
        # Create user profile as average of interacted book features
        user_read_vectors = [self.book_vectors[idx] for idx in interacted_indices]
        user_profile = average_sparse_vectors(user_read_vectors)
        
        # Calculate custom sparse cosine similarity with all books
        sim_scores = np.zeros(len(self.book_vectors))
        for i, book_vec in enumerate(self.book_vectors):
            if i in interacted_indices_set:
                sim_scores[i] = -1.0 # Exclude read books
            else:
                sim_scores[i] = cosine_similarity_sparse(user_profile, book_vec)
        
        # Get top indices
        top_indices = np.argpartition(sim_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-sim_scores[top_indices])]
        
        return self.books[top_indices].tolist()

class Evaluator:
    def __init__(self, data_processor):
        self.dp = data_processor
        self.merged_data = pd.merge(
            self.dp.interactions, 
            self.dp.chapters[['chapter_id', 'chapter_sequence_no']], 
            on='chapter_id', 
            how='left'
        )

    def evaluate_new_book(self, num_users_to_eval=1000, top_k=10):
        print(f"\nEvaluating New Book Recommender on {num_users_to_eval} users (HR@{top_k})...")
        book_counts = self.merged_data.groupby('user_id')['book_id'].nunique()
        valid_users = book_counts[book_counts > 1].index
        
        np.random.seed(42)
        sampled_users = np.random.choice(valid_users, min(num_users_to_eval, len(valid_users)), replace=False)
        
        new_book_rec = NewBookRecommender(self.dp.book_metadata)
        new_book_rec.prepare_popularity_baseline(self.merged_data)
        
        hits = 0
        total = 0
        
        for i, user in enumerate(sampled_users):
            user_data = self.merged_data[self.merged_data['user_id'] == user]
            user_books = user_data['book_id'].unique().tolist()
            
            held_out_book = np.random.choice(user_books)
            train_books = [b for b in user_books if b != held_out_book]
            
            recs = new_book_rec.recommend(train_books, top_k=top_k)
            
            if held_out_book in recs:
                hits += 1
            total += 1
            
            if (i+1) % 200 == 0:
                print(f"   Processed {i+1} users...")

        hr = hits / total if total > 0 else 0
        print(f"Evaluation Complete. HR@{top_k}: {hr:.4f}")
        return hr

if __name__ == "__main__":
    dp = DataProcessor()
    dp.load_data()
    
    # Run simple evaluation
    evaluator = Evaluator(dp)
    evaluator.evaluate_new_book(num_users_to_eval=500, top_k=10)
    
    # Demonstrate on single user
    print("\n--- Example Demonstration ---")
    user_id = evaluator.merged_data['user_id'].iloc[0]
    user_data = evaluator.merged_data[evaluator.merged_data['user_id'] == user_id]
    user_books = user_data['book_id'].unique().tolist()
    
    print(f"User {user_id} has interacted with books: {user_books}")
    
    cr_rec = ContinueReadingRecommender(dp.chapters)
    cr_recs = cr_rec.recommend(user_data)
    print(f"Continue Reading Recommendations (Book -> Next Chapter Sequence):")
    for book, next_chap in cr_recs.items():
        print(f"  - Book {book}: Chapter {next_chap}")
        
    nbr = NewBookRecommender(dp.book_metadata)
    nbr.prepare_popularity_baseline(evaluator.merged_data)
    new_recs = nbr.recommend(user_books, top_k=5)
    print(f"New Book Recommendations (Top 5 Book IDs): {new_recs}")
