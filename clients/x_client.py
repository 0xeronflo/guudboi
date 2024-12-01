import os
import tweepy
import time
from datetime import datetime


class XClient:
    def __init__(self):
        print("Access keys being used:")
        print("BROWSER_TOKEN:", os.getenv("X_BEARER_TOKEN"))
        print("CONSUMER_KEY:", os.getenv("X_API_KEY"))
        print("ACCESS_TOKEN:", os.getenv("X_ACCESS_TOKEN"))
        self.client = tweepy.Client(
            bearer_token=os.getenv("X_BEARER_TOKEN"),
            consumer_key=os.getenv("X_API_KEY"),
            consumer_secret=os.getenv("X_API_SECRET"),
            access_token=os.getenv("X_ACCESS_TOKEN"),
            access_token_secret=os.getenv("X_ACCESS_SECRET"),
            wait_on_rate_limit=True,
        )

    def fetch_tweets(self, list_id, max_results=5, generate_description=None):
        """
        Fetch tweets from a Twitter list, optionally generating media descriptions.
        :param list_id: ID of the Twitter list to fetch tweets from.
        :param max_results: Maximum number of tweets to fetch.
        :param generate_description: Optional callback to generate descriptions for media URLs.
        :return: List of tweets with enriched metadata for AI analysis.
        """
        try:
            print(
                f"\n### Fetching Up to {max_results} Tweets from List ID: {list_id} ###")

            response = self.client.get_list_tweets(
                id=list_id,
                max_results=max_results,
                tweet_fields=["created_at", "text", "author_id",
                              "note_tweet", "referenced_tweets", "public_metrics"],
                expansions=["author_id", "attachments.media_keys",
                            "referenced_tweets.id", "referenced_tweets.id.author_id"],
                media_fields=["media_key", "url", "type"],
                user_fields=["username"],
            )

            tweets = []
            if response.data:
                for tweet in response.data:
                    original_id = tweet.id
                    ref_tweet = None
                    ref_id = None

                    if tweet.referenced_tweets:
                        ref_id = tweet.referenced_tweets[0]["id"]
                        try:
                            ref_response = self.client.get_tweet(
                                id=ref_id,
                                tweet_fields=[
                                    "created_at", "text", "author_id", "note_tweet", "public_metrics"],
                                expansions=["author_id",
                                            "attachments.media_keys"],
                                media_fields=["media_key", "url", "type"],
                                user_fields=["username"],
                            )
                            ref_tweet = ref_response.data
                        except tweepy.TweepyException as e:
                            print(
                                f"Error fetching referenced tweet with ID {ref_id}: {e}")

                    chosen_tweet = ref_tweet if ref_tweet else tweet
                    chosen_id = ref_tweet["id"] if ref_tweet else original_id
                    full_text = chosen_tweet.get("note_tweet", {}).get(
                        "text", chosen_tweet["text"])
                    author_id = chosen_tweet["author_id"]
                    attachments = chosen_tweet.get("attachments", {})
                    public_metrics = chosen_tweet.get("public_metrics", {})
                    created_at = chosen_tweet.get("created_at")
                    created_at_str = created_at.isoformat() if created_at else None

                    author_username = next(
                        (user["username"] for user in response.includes.get(
                            "users", []) if user["id"] == author_id),
                        "Unknown"
                    )

                    media_urls = [
                        media["url"]
                        for media in response.includes.get("media", [])
                        if attachments and media.get("media_key") in attachments.get("media_keys", []) and media.get("type") == "photo"
                    ]

                    # Use the provided callback for image descriptions, if any
                    image_descriptions = [generate_description(
                        url) for url in media_urls] if generate_description else []

                    tweet_info = {
                        "tweet_id": chosen_id,
                        "original_tweet_id": original_id,
                        "author_username": author_username,
                        "text": full_text,
                        "media_urls": media_urls,
                        "image_descriptions": image_descriptions,
                        "engagement_metrics": public_metrics,
                        "created_at": created_at_str,
                    }
                    tweets.append(tweet_info)

            print(f"\n### Retrieved {len(tweets)} Tweets ###")
            return tweets
        except tweepy.TweepyException as e:
            print(f"Error fetching tweets: {e}")
            return []

    def post_quote_tweet(self, quote_tweet_id, response_text, thread=[], max_retries=4, retry_delay=15):
        """
        Post a quote tweet response, optionally including a thread.
        If the thread list is empty, only the main tweet will be posted.
        Retries on connection-related errors for each tweet individually.
        :param quote_tweet_id: The ID of the tweet to quote.
        :param response_text: The text of the main quote tweet.
        :param thread: List of additional tweets to post as a thread (optional).
        :param max_retries: Maximum number of retries for posting each tweet.
        :param retry_delay: Delay in seconds between retries.
        :return: Tuple containing the main tweet info and thread tweets info, or None if the main tweet fails.
        """
        # Helper to post a single tweet with retry
        def post_tweet_with_retry(text, in_reply_to=None, retries=max_retries, delay=retry_delay):
            for attempt in range(retries):
                try:
                    if in_reply_to:
                        response = self.client.create_tweet(
                            text=text, in_reply_to_tweet_id=in_reply_to
                        )
                    else:
                        response = self.client.create_tweet(
                            text=text, quote_tweet_id=quote_tweet_id
                        )
                    tweet_id = response.data.get("id")
                    return {
                        "tweet_id": tweet_id,
                        "text": text,
                        "created_at": datetime.utcnow().isoformat(),
                        "author_username": "ai_meme_review",
                    }
                except (ConnectionError, tweepy.TweepyException) as e:
                    print(
                        f"Error posting tweet (attempt {attempt + 1}/{retries}): {e}")
                    if attempt < retries - 1:
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print("Max retries reached. Skipping this tweet.")
            return None

        # Post the main quote tweet
        print("Posting main quote tweet...")
        main_tweet = post_tweet_with_retry(response_text)
        if not main_tweet:
            print("Failed to post the main tweet. Aborting entire response.")
            return None  # Abort if the main tweet fails

        # Post thread tweets if any
        thread_tweets = []
        if thread:
            in_reply_to_tweet_id = main_tweet["tweet_id"]
            for tweet_text in thread:
                print(f"Posting thread tweet: {tweet_text}")
                thread_tweet = post_tweet_with_retry(
                    tweet_text, in_reply_to=in_reply_to_tweet_id)
                if thread_tweet:
                    thread_tweets.append(thread_tweet)
                    # Update the reply chain
                    in_reply_to_tweet_id = thread_tweet["tweet_id"]
                else:
                    print(
                        "Failed to post a thread tweet. Continuing with remaining tweets.")

        return main_tweet, thread_tweets

    def post_reply(self, in_reply_to_tweet_id, reply_text, max_retries=4, retry_delay=15):
        """
        Post a direct reply to a tweet.
        Retries on connection-related errors.
        :param in_reply_to_tweet_id: The ID of the tweet to reply to.
        :param reply_text: The text of the reply.
        :return: Dictionary containing the reply tweet info or None if failed.
        """
        print(f"Posting reply to tweet ID {in_reply_to_tweet_id}...")
        for attempt in range(max_retries):
            try:
                response = self.client.create_tweet(
                    text=reply_text, in_reply_to_tweet_id=in_reply_to_tweet_id
                )
                tweet_id = response.data.get("id")
                return {
                    "tweet_id": tweet_id,
                    "text": reply_text,
                    "created_at": datetime.utcnow().isoformat(),
                    "author_username": "ai_meme_review",
                }
            except (ConnectionError, tweepy.TweepyException) as e:
                print(
                    f"Error posting reply (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Reply failed.")
        return None
