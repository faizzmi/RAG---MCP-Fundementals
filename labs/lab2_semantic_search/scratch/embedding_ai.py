import logging
from openai import OpenAI, RateLimitError, AuthenticationError, APIError
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=config.api_key)

sentences = [
    "Dogs are allowed in the office on Fridays.",
    "Pets can come to work on Furry Fridays.",
    "Remote work policy allows 3 days from home.",
]


def get_openai_embedding(text_list):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text_list
        )
        return [item.embedding for item in response.data]

    except RateLimitError as e:
        logger.error("Quota/rate limit hit — check billing at platform.openai.com. Details: %s", e)
        return None

    except AuthenticationError as e:
        logger.error("Invalid or missing API key. Check your .env file. Details: %s", e)
        return None

    except APIError as e:
        logger.error("OpenAI API returned an error: %s", e)
        return None

    except Exception as e:
        logger.exception("Unexpected error while generating embeddings")
        return None


embeddings = get_openai_embedding(sentences)

if embeddings is not None:
    logger.info("%d embeddings generated, each with %d dimensions", len(embeddings), len(embeddings[0]))
else:
    logger.warning("No embeddings were generated, see error above")