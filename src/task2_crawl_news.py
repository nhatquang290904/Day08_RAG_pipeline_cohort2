"""
Task 2 - Crawl news articles about Vietnamese artists related to drug cases.

Outputs one JSON file per article into data/landing/news/ with:
url, title, date_crawled, source_domain, content_markdown.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://tuoitre.vn/nu-nguoi-mau-an-tay-ca-si-chi-dan-bi-dieu-tra-nghi-lien-quan-den-ma-tuy-20241110100605268.htm",
    "https://thanhnien.vn/ca-si-chi-dan-bi-cong-an-dieu-tra-nghi-lien-quan-ma-tuy-185241110101919332.htm",
    "https://vtv.vn/phap-luat/bat-ca-si-chi-dan-nguoi-mau-an-tay-tiktoker-truc-phuong-do-lien-quan-ma-tuy-20241114123427363.htm",
    "https://dantri.com.vn/phap-luat/cong-an-tphcm-doc-lenh-bat-co-tien-truc-phuong-nguoi-mau-an-tay-20241114143106380.htm",
    "https://cuoi.tuoitre.vn/loat-nghe-si-viet-tieu-tan-su-nghiep-vi-ma-tuy-20241114142620463.htm",
]


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _crawl_with_requests(url: str) -> dict:
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    html = response.content.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    title_tag = (
        soup.find("meta", property="og:title")
        or soup.find("meta", attrs={"name": "title"})
    )
    if title_tag and title_tag.get("content"):
        title = title_tag["content"]
    elif soup.find("h1"):
        title = soup.find("h1").get_text(" ", strip=True)
    else:
        title = urlparse(url).path.rsplit("/", 1)[-1]

    selectors = [
        "article",
        ".detail-content",
        ".article-content",
        ".singular-content",
        ".dt-news__content",
        ".contentdetail",
        ".fck_detail",
        ".cms-body",
        ".edittor-content",
    ]
    content_root = None
    for selector in selectors:
        content_root = soup.select_one(selector)
        if content_root:
            break
    if content_root is None:
        content_root = soup.body or soup

    paragraphs = [
        _clean_text(p.get_text(" ", strip=True))
        for p in content_root.find_all(["p", "h1", "h2"])
    ]
    paragraphs = [p for p in paragraphs if len(p) > 30]
    content = "\n\n".join(dict.fromkeys(paragraphs))

    return {
        "url": url,
        "title": _clean_text(title),
        "date_crawled": datetime.now().isoformat(),
        "source_domain": urlparse(url).netloc,
        "content_markdown": content,
    }


async def crawl_article(url: str) -> dict:
    """Crawl one article and return metadata plus article text."""
    return _crawl_with_requests(url)


async def crawl_all() -> None:
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
