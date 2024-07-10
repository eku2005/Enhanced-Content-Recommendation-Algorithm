from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Example function to create user profiles based on interactions
def create_user_profiles(user_interactions, content_metadata):
    # Aggregate user interactions
    user_profiles = user_interactions.groupby('user_id').agg(list).reset_index()
    
    # Merge with content metadata
    user_profiles = user_profiles.merge(content_metadata, on='content_id')
    
    return user_profiles

user_profiles = create_user_profiles(user_interactions, content_metadata)

# Example function to cluster users based on their interaction patterns
def cluster_users(user_profiles):
    # Convert tags to a single string for each user
    user_profiles['tags'] = user_profiles['tags'].apply(lambda x: ' '.join(x))
    
    # Use TF-IDF to vectorize tags
    vectorizer = TfidfVectorizer()
    user_tags_matrix = vectorizer.fit_transform(user_profiles['tags'])
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_profiles['cluster'] = kmeans.fit_predict(user_tags_matrix)
    
    return user_profiles

clustered_user_profiles = cluster_users(user_profiles)
