
import requests
from bs4 import BeautifulSoup


def read_website(url: str) -> str:
    """Download a web page and extract its readable text."""

    headers = {
        "User-Agent": "PorscheResearchBot/1.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove elements that usually aren't useful as article text
    for element in soup.select(
        "script, style, nav, footer, header"
    ):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)

    return text


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Porsche_918_Spyder"

    try:
        article_text = read_website(url)

        print("Website downloaded successfully!")
        print("Characters extracted:", len(article_text))
        print("\nFirst 1500 characters:\n")
        print(article_text[:1500])

    except requests.RequestException as error:
        print("Could not download website:", error)
