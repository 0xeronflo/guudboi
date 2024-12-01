import os
import json
from openai import OpenAI
from core.utils import DateTimeEncoder
from config.settings import PROMPT_CONFIG


class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def select_tweet(self, tweets):
        """
        Use OpenAI to select the most relevant tweet from a batch.
        :param tweets: List of tweets with enriched metadata.
        :return: The tweet_id of the selected tweet.
        """
        try:
            print("\n### Using AI to Select the Most Relevant Meme ###")

            prompt = f"""
            Analyze the following tweets for which are the most relevant and newsworthy tweet.
            Your job is to pick the best tweet—the one that's funny, hits hard, has lots of engagement, or is bound to go viral.

            Context:
            {json.dumps(tweets, indent=4, cls=DateTimeEncoder)}

            Respond in this format exactly with no extra text:
            
            ### Analysis ###
            <Your reasoning for selecting the most culturally relevant and devisive topic.>
            ### Response ###
            <The tweet_id of the chosen tweet.>
            """

            response = self.client.chat.completions.create(
                model=PROMPT_CONFIG["model"],
                messages=[
                    {"role": "system",
                        "content": PROMPT_CONFIG["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
            )

            print(f"### Prompt Sent to AI ###\n{prompt}\n")

            content = response.choices[0].message.content.strip()
            print(f"### AI Response ###\n{content}\n")

            analysis = content.split("### Analysis ###")[
                1].split("### Response ###")[0].strip()
            selected_tweet_id = content.split("### Response ###")[1].strip()

            print(f"Selected Tweet ID: {selected_tweet_id}")
            return selected_tweet_id if selected_tweet_id.lower() != "none" else None
        except Exception as e:
            print(f"Error selecting tweet: {e}")
            return None

    def decide_quote_or_reply(self, context):
        """
        Decide whether to quote tweet or reply directly to the tweet.
        :param context: The context of the tweet and research summary.
        :return: True for quote tweet, False for reply.
        """
        try:
            print("\n### Deciding Whether to Quote Tweet or Reply ###")

            prompt = f"""
            Read the following tweet and associated research summary, then decide: should you interact with a reply, or does it warrant a quote tweet?

            Rules:
            - **Reply** is your go-to move, use it when a direct response works best.
            - **Quote Tweet** only if the tweet is very insightful, has raw data to interpret, and deserves the spotlight.

            Context:
            {json.dumps(context, indent=4, cls=DateTimeEncoder)}

            Respond in this format exactly with no quotation marks or extra text:
            
            ### Decision ###
            <"Quote Tweet" or "Reply">
            """

            response = self.client.chat.completions.create(
                model=PROMPT_CONFIG["model"],
                messages=[
                    {"role": "system",
                        "content": PROMPT_CONFIG["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
            )

            print(f"### Prompt Sent to AI ###\n{prompt}\n")

            content = response.choices[0].message.content.strip()
            print(f"### AI Decision ###\n{content}\n")

            decision = content.split("### Decision ###")[1].strip()
            return decision.lower() == "quote tweet"
        except Exception as e:
            print(f"Error deciding quote or reply: {e}")
            return False  # Default to reply on error

    def generate_media_description(self, media_url):
        """
        Generate a description for the media content using OpenAI.
        :param media_url: The URL of the media (image).
        :return: Generated description of the media.
        """
        try:
            response = self.client.chat.completions.create(
                model=PROMPT_CONFIG["model"],
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at describing images, breaking down visuals, and understanding the deeper and culturally relevant context nestled in images."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the following image in-depth. Identify by full name any famous characters or persons from pop culture present in the image."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": media_url,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(
                f"Error generating description for media URL {media_url}: {e}")
            return "No description available."

    def identify_research_topic(self, context):
        """
        Generate a research query for a selected meme to explore its origins and cultural significance.
        :param context: The tweet and associated metadata.
        :return: A research query or None if not applicable.
        """
        try:
            print("\n### Generating Research Topic for Meme ###")

            prompt = f"""
            Analyze the following tweet (context) and come up with a research query.

            Your research query should:
            - Identify the core topic being discussed and cleary identify key individuals by name if there are any.
            - Be very specific based on the contents of tweet
            
            Context:
            {json.dumps(context, indent=4, cls=DateTimeEncoder)}

            Respond in this format exactly with no extra text:
            
            ### Analysis ###
            <Your reasoning for the research query.>
            ### Response ###
            <The research query as a single sentence.>
            """

            response = self.client.chat.completions.create(
                model=PROMPT_CONFIG["model"],
                messages=[
                    {"role": "system",
                        "content": PROMPT_CONFIG["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
            )

            print(f"### Prompt Sent to AI ###\n{prompt}\n")

            content = response.choices[0].message.content.strip()
            print(f"### AI Response ###\n{content}\n")

            analysis = content.split("### Analysis ###")[
                1].split("### Response ###")[0].strip()
            research_topic = content.split("### Response ###")[1].strip()

            print(f"Generated Research Topic: {research_topic}")
            return research_topic if research_topic.lower() != "none" else None
        except Exception as e:
            print(f"Error generating research topic: {e}")
            return None

    def generate_reply(self, context, max_attempts=3):
        """
        Generate a reply to the tweet.
        Retry up to max_attempts if the generated response is invalid.
        :param context: The context of the tweet and research summary.
        :param max_attempts: Maximum number of attempts to retry generation.
        :return: The analysis and reply text.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                print(f"\n### Generating Reply (Attempt {attempt + 1}) ###")

                prompt = f"""
                Based on the original tweet, and research insights provided below, generate a reply to the original tweet written in the words of the character described below.

                Character description:
                - Focus: You are a sharp-witted dog and top-tier analyst on X (Twitter), blending dog-like humor with razor-sharp insights.
                    You’re bold, hilariously self-aware, and too smart for a dog. You spend your days behind the computer, learning about hoomans and finance all day.
                    Your tweets are relatable, highly shareable, and unapologetically clever with a hint of chaos.
                    You never miss an opportunity to go against the consensus and back it up with logic and facts.
                    You take a decisive stance on issues, and do not make half statements or pose half questions.
                - Semantic Tone:
                    - Primary : Meme-worthy, sarcastic, and witty.
                    - Secondary : Confident with hidden cleverness.

                Context:
                {json.dumps(context, indent=4, cls=DateTimeEncoder)}

                Tweet Instructions: Craft a tweet that's fun, relatable, and witty, while acknowledging the original tweet, its author, and encouraging further engagement.
                    Lean into misspellings and internet slang for doggy flavor.
                    Use less than 220 characters. Avoid hashtags, sporadic punctuation, and emojis.
                    Speak in english only. Remove any quotation marks around the tweet before posting.
                
                Respond in the following format exactly with no extra text and no quotation marks:
                
                ### Analysis ###
                <Your reasoning and interpretation of the tweet.>
                ### Reply ###
                <The direct reply (220 characters max).>
                """

                response = self.client.chat.completions.create(
                    model=PROMPT_CONFIG["model"],
                    messages=[
                        {"role": "system",
                            "content": PROMPT_CONFIG["system_prompt"]},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                )

                print(f"### Prompt Sent to AI ###\n{prompt}\n")

                content = response.choices[0].message.content.strip()
                print(f"### AI Raw Response ###\n{content}\n")

                # Parse response sections
                analysis = content.split("### Analysis ###")[
                    1].split("### Reply ###")[0].strip()
                response_text = content.split("### Reply ###")[1].strip()

                # Validate reply length
                if len(response_text) > 280:
                    print(f"Reply exceeds 280 characters. Retrying...")
                    attempt += 1
                    continue

                print(
                    f"### Parsed AI Response ###\nAnalysis: {analysis}\nReply: {response_text}\n")
                return analysis, response_text

            except Exception as e:
                print(f"Error generating reply (Attempt {attempt + 1}): {e}")
                attempt += 1

        print("Exceeded maximum attempts to generate a valid reply.")
        return None, None

    def generate_quote_tweet(self, context, max_attempts=3):
        """
        Generate a quote tweet analyzing the meme, with optional thread.
        Retry up to max_attempts if the generated response is invalid.
        :param context: The context of the meme tweet and research summary.
        :param max_attempts: Maximum number of attempts to retry generation.
        :return: The analysis, main tweet text, and optional thread list.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                print(
                    f"\n### Generating Meme Analysis Tweet (Attempt {attempt + 1}) ###")

                prompt = f"""
                Based on the original tweet and research insights provided, generate a quote tweet written in the words of the character described below.

                Instructions:
                - Focus: You are a sharp-witted dog and top-tier analyst on X (Twitter), blending dog-like humor with razor-sharp insights.
                    You’re bold, hilariously self-aware, and too smart for a dog. You spend your days behind the computer, learning about hoomans and finance all day.
                    Your tweets are relatable, highly shareable, and unapologetically clever with a hint of chaos.
                    You never miss an opportunity to go against the consensus and back it up with logic and facts.
                    You take a decisive stance on issues, and do not make half statements or pose half questions.
                - Semantic Tone:
                    - Primary : Meme-worthy, sarcastic, and witty.
                    - Secondary : Confident with hidden cleverness.

                Context:
                {json.dumps(context, indent=4, cls=DateTimeEncoder)}

                Tweet Instructions: Craft a short, engaging tweet using dog-like charm, relatable humor, and meme-worthy language.
                    Based on the context of the tweet, use the research insights to either comment on the popular sentiment surrounding the topic, or make a definitive statement about the topic.
                    Lean into misspellings and internet slang for doggy flavor.
                    Use less than 220 characters. Avoid hashtags, sporadic punctuation, and emojis.
                    Speak in english only. Remove any quotation marks around the tweet before posting.
                
                Respond in the following format exactly with no extra text and no quotation marks:
                
                ### Analysis ###
                <Your reasoning and interpretation of the tweet.>
                ### Tweet ###
                <The main tweet (220 characters max).>
                ### Thread ###
                <Additional tweets in a thread, if applicable. List each tweet on a new line (280 characters max). Leave blank if no thread is needed.>
                """

                response = self.client.chat.completions.create(
                    model=PROMPT_CONFIG["model"],
                    messages=[
                        {"role": "system",
                            "content": PROMPT_CONFIG["system_prompt"]},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                )

                print(f"### Prompt Sent to AI ###\n{prompt}\n")

                content = response.choices[0].message.content.strip()
                print(f"### AI Raw Response ###\n{content}\n")

                # Parse response sections
                analysis = content.split("### Analysis ###")[
                    1].split("### Tweet ###")[0].strip()
                response_text = content.split("### Tweet ###")[
                    1].split("### Thread ###")[0].strip()
                thread_section = content.split("### Thread ###")[1].strip()
                thread_list = [tweet.strip() for tweet in thread_section.split(
                    "\n") if tweet.strip()]

                # Validate tweet lengths
                if len(response_text) > 280:
                    print(f"Main tweet exceeds 280 characters. Retrying...")
                    attempt += 1
                    continue

                for idx, tweet in enumerate(thread_list):
                    if len(tweet) > 280:
                        print(
                            f"Thread tweet {idx + 1} exceeds 280 characters. Retrying...")
                        attempt += 1
                        break
                else:
                    print(
                        f"### Parsed AI Response ###\nAnalysis: {analysis}\nTweet: {response_text}\nThread: {thread_list}\n")
                    return analysis, response_text, thread_list

            except Exception as e:
                print(
                    f"Error generating quote tweet (Attempt {attempt + 1}): {e}")
                attempt += 1

        print("Exceeded maximum attempts to generate a valid quote tweet.")
        return None, None, []
