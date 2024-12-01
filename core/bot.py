from datetime import datetime
import time
import random
from core.utils import log_header
from config.settings import LIST_ID, CYCLE_LENGTH, MAX_RESULTS
from clients.x_client import XClient
from clients.openai_client import OpenAIClient
from clients.perplexity_client import PerplexityClient

# Initialize global variables
processed_tweet_ids = set()


def setup_clients():
    """Initialize all clients and ensure the schema is set up."""
    x_client = XClient()
    openai_client = OpenAIClient()
    perplexity_client = PerplexityClient()
    return x_client, openai_client, perplexity_client


def process_fetched_tweets(fetched_tweets, openai_client):
    """Process unprocessed tweets by enriching them and selecting one."""
    # Filter out already processed tweets
    unprocessed_tweets = [
        tweet for tweet in fetched_tweets if tweet["tweet_id"] not in processed_tweet_ids
    ]

    if not unprocessed_tweets:
        print("No unprocessed tweets available. Skipping this cycle.")
        return None, None

    # Pass unprocessed tweets to AI for selection
    selected_tweet_id = openai_client.select_tweet(unprocessed_tweets)
    if not selected_tweet_id:
        print("AI did not select any tweet. Skipping this cycle.")
        return None, None

    # Locate the selected tweet
    selected_tweet = next(
        (tweet for tweet in unprocessed_tweets if str(
            tweet["tweet_id"]) == selected_tweet_id),
        None
    )
    return selected_tweet, unprocessed_tweets


def handle_selected_tweet(selected_tweet, openai_client, perplexity_client, x_client):
    """Handle the selected tweet by deciding whether to quote tweet or reply directly."""
    if not selected_tweet:
        print("No tweet selected for processing.")
        return

    # Generate response context
    context = {"tweet": selected_tweet}
    research_topic = openai_client.identify_research_topic(context)
    research_summary = (
        perplexity_client.research_topic(
            research_topic) if research_topic else None
    )
    context["research_summary"] = research_summary or "No additional insights available."

    # Decide on the type of response
    is_quote_tweet = openai_client.decide_quote_or_reply(context)

    if is_quote_tweet:
        # Generate a quote tweet
        analysis, response_text, thread = openai_client.generate_quote_tweet(
            context
        )
    else:
        # Generate a reply
        analysis, response_text = openai_client.generate_reply(context)

    if not response_text:
        print("Failed to generate a response. Skipping this tweet.")
        return

    # Post the response
    if is_quote_tweet:
        post_result = x_client.post_quote_tweet(
            quote_tweet_id=selected_tweet["tweet_id"],
            response_text=response_text,
            thread=thread,
        )
    else:
        post_result = x_client.post_reply(
            in_reply_to_tweet_id=selected_tweet["tweet_id"],
            reply_text=response_text,
        )

    if post_result:
        # Mark the selected tweet as processed
        processed_tweet_ids.add(selected_tweet["tweet_id"])


def main_cycle(x_client, openai_client, perplexity_client):
    """Main cycle for analyzing, generating, and posting tweets."""
    while True:
        try:
            log_header(
                f"Starting new cycle at {datetime.utcnow().isoformat()}")

            # Fetch tweets and generate media descriptions using OpenAIClient
            fetched_tweets = x_client.fetch_tweets(
                list_id=LIST_ID,
                max_results=MAX_RESULTS,
                # Pass the method to generate media descriptions as a callback
                generate_description=openai_client.generate_media_description
            )
            if not fetched_tweets:
                print("No tweets fetched from the list. Skipping this cycle.")
                continue

            # Process fetched tweets
            selected_tweet, _ = process_fetched_tweets(
                fetched_tweets, openai_client
            )

            # Handle the selected tweet
            handle_selected_tweet(
                selected_tweet, openai_client, perplexity_client, x_client
            )

        except Exception as e:
            print(f"Error during cycle: {e}")

        # Add random variance to the cycle wait time
        variance = random.uniform(-0.10, 0.10)  # ±10% variance
        # Ensure a minimum of 1 second
        cycle_time = CYCLE_LENGTH + int(CYCLE_LENGTH * variance)
        print(f"\nCycle complete. Waiting for {cycle_time} seconds...\n")
        time.sleep(cycle_time)


if __name__ == "__main__":
    # Initialize clients
    x_client, openai_client, perplexity_client = setup_clients()

    # Start the main cycle
    main_cycle(x_client, openai_client, perplexity_client)