import pandas as pd

# Simulated user interaction data
user_interactions = pd.DataFrame({
    'user_id': [1, 2, 1, 3, 2, 1],
    'content_id': [101, 102, 103, 101, 103, 104],
    'interaction_type': ['like', 'share', 'comment', 'like', 'like', 'comment'],
    'timestamp': pd.to_datetime(['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05', '2024-07-06'])
})

# Simulated content metadata
content_metadata = pd.DataFrame({
    'content_id': [101, 102, 103, 104],
    'title': ['Post 1', 'Post 2', 'Post 3', 'Post 4'],
    'tags': [['tag1', 'tag2'], ['tag2', 'tag3'], ['tag1', 'tag3'], ['tag1', 'tag2', 'tag3']]
})

# Simulated social connections
social_connections = pd.DataFrame({
    'user_id': [1, 2, 3, 1],
    'friend_id': [2, 1, 1, 3]
})

# Merge data for further analysis
merged_data = user_interactions.merge(content_metadata, on='content_id')
