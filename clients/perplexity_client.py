import os
from openai import OpenAI


class PerplexityClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )

    def research_topic(self, topic):
        """
        Use OpenAI (via Perplexity API) to research a meme-related topic.
        :param topic: The topic to research.
        :return: The AI-generated research summary.
        """
        try:
            print(f"\n### Researching Topic: {topic} ###")

            # Define the role and user input for the conversation
            prompt = f"""
            Provide concise, detailed research on this topic and any notable individuals mentioned, designed to inform an analyst / commentator on X (twitter), ensuring you consider the latest developments first: {topic}.
            Break down the analysis into the following subtopics, ensuring brevity and clarity:
            - Brief overview of the topic
            - Latest developments in the news
            - Financial/economic implications
            - Popular consensus
            - Key arguments for and against the consensus, and their logic
            - Notable individuals and their relevance in the context of the tweet. Reply 'none' if no notable individuals.
            - Conclusion
            """

            # Send the request to the Perplexity API
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[
                    {"role": "system", "content": "You are an analyst whose mission is to spark debate by blending meme culture with biting wit and cleverness."},
                    {"role": "user", "content": prompt},
                ],
            )

            # Extract and return the response content
            research_summary = response.choices[0].message.content.strip()
            print(f"\n### Research Summary ###\n{research_summary}\n")
            return research_summary

        except Exception as e:
            print(f"Error while researching topic: {e}")
            return None
