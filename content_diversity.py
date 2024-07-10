import random

# Example function to ensure diversity in recommendations
def recommend_content(user_id, user_profiles, content_metadata, model, top_n=10):
    user_cluster = user_profiles[user_profiles['user_id'] == user_id]['cluster'].values[0]
    cluster_users = user_profiles[user_profiles['cluster'] == user_cluster]['user_id'].values

    # Get all items
    all_items = content_metadata['content_id'].values

    # Predict scores for all items for the given user
    user_vector = np.array([user_id] * len(all_items))
    item_vector = np.array(all_items)
    predictions = model.predict(np.array([user_vector, item_vector]).T)
    
    # Rank items by predicted score
    ranked_items = [item for _, item in sorted(zip(predictions, all_items), reverse=True)]
    
    # Ensure diversity by adding random items from other clusters
    recommended_items = ranked_items[:top_n//2]
    other_items = [item for item in all_items if item not in recommended_items]
    recommended_items.extend(random.sample(list(other_items), top_n//2))
    
    return recommended_items

recommended_items = recommend_content(1, clustered_user_profiles, content_metadata, model)
