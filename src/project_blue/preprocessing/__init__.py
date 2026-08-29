import pandas as pd
import re
from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_languages().build()

def verify_language(text: str, langs_tag: list, detector_object) -> bool:
    """"""
    if "en" in langs_tag:
        # Returns a list with languages probabilities
        result = detector_object.compute_language_confidence_values(text)

        if result[0].language == Language.ENGLISH:
            return True
        else:
            return False
    else:
        return False


def normalize_text(text: str) -> str:
    """"""
    # Changing all Urls to [URL]
    text = re.sub(r"https?://\S+", "[URL]", text)

    # Changing all mentions in post to "[MENTION]"
    text = re.sub(r"@\S+", "[MENTION]", text)

    # Removing all hashtags symbol before words
    text = re.sub(r"#", "", text)

    return text

if __name__ == "__main__":
    print(normalize_text("Check this out @john_doe https://example.com #MachineLearning is cool"))