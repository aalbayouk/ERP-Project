import praw
import logging
import pandas as pd

# Configure PRAW with your Reddit API credentials
reddit = praw.Reddit(client_id='XXXXXXXXX',
                     client_secret='XXXXXXXX',
                     user_agent='XXXXXXXXXXX')
# Function to retrieve the URL of a post using its post_id
def get_post_url_by_id(post_id):
    try:
        submission = reddit.submission(id=post_id)
        logging.info(f"Found URL for post_id '{post_id}': {submission.url}")
        return submission.url
    except Exception as e:
        logging.error(f"Error fetching URL for post_id '{post_id}': {e}")
        return None

# Load the dataset
df = pd.read_csv('{add directory}/url_extraction_dfs.csv', low_memory=False, encoding='utf-8-sig')

# Apply the URL extraction function using post_id
df['url'] = df['post_id'].apply(get_post_url_by_id)

# Check if URLs were found
print("Number of URLs found:", df['url'].notnull().sum())

# Log missing URLs for further inspection
missing_urls = df[df['url'].isnull()]['post_id']
missing_urls.to_csv('{add directory}/missing_urls.csv', index=False)

# Save the updated DataFrame to a new CSV file
df.to_csv('{add directory}/post_id_with_url_extracted.csv', index=False)
