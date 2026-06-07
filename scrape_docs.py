#!/usr/bin/env python3
"""Scrape documentation sites into local markdown folders."""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
USER_AGENT = "ideation-docs-scraper/1.0 (+local mirror)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def fetch(url: str, *, retries: int = 3, timeout: int = 60) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = SESSION.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def parse_llms_index(text: str, base_url: str) -> list[dict[str, str]]:
    pages: list[dict[str, str]] = []
    for line in text.splitlines():
        match = re.match(r"- \[(?P<title>[^\]]+)\]\((?P<url>[^)]+)\)(?:: (?P<desc>.*))?", line)
        if not match:
            continue
        url = match.group("url")
        if not url.endswith(".md"):
            continue
        pages.append(
            {
                "title": match.group("title"),
                "url": urljoin(base_url, url),
                "description": match.group("desc") or "",
            }
        )
    return pages


def url_to_local_path(url: str, prefix: str) -> Path:
    parsed = urlparse(url)
    rel = parsed.path.lstrip("/")
    if rel.endswith(".md"):
        rel = rel[:-3] + ".md"
    return Path(prefix) / rel


def scrape_llms_site(
    *,
    name: str,
    index_url: str,
    base_url: str,
    output_dir: Path,
    workers: int = 12,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    index_text = fetch(index_url).text
    (output_dir / "llms.txt").write_text(index_text, encoding="utf-8")

    pages = parse_llms_index(index_text, base_url)
    results: list[dict] = []
    failures: list[dict] = []

    def download(page: dict) -> dict:
        local_path = pages_dir / url_to_local_path(page["url"], "pages").relative_to("pages")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        content = fetch(page["url"]).text
        header = (
            f"> Source: {page['url']}\n"
            f"> Title: {page['title']}\n"
        )
        if page["description"]:
            header += f"> Description: {page['description']}\n"
        local_path.write_text(header + "\n" + content, encoding="utf-8")
        return {
            "title": page["title"],
            "url": page["url"],
            "path": str(local_path.relative_to(output_dir)),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download, page): page for page in pages}
        for future in as_completed(futures):
            page = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failures.append({"url": page["url"], "error": str(exc)})

    manifest = {
        "source": index_url,
        "docs_site": base_url,
        "total_pages": len(pages),
        "successful": len(results),
        "failed": len(failures),
        "pages_dir": "pages",
        "index": "llms.txt",
        "failures": failures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{name}] {len(results)}/{len(pages)} pages saved to {output_dir}")
    if failures:
        print(f"[{name}] {len(failures)} failures")
        for item in failures[:5]:
            print(f"  - {item['url']}: {item['error']}")
    return manifest


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for selector in ("script", "style", "nav", "header", "footer", "aside"):
        for tag in soup.select(selector):
            tag.decompose()

    article = soup.select_one("article.article") or soup.select_one("main") or soup.body
    if article is None:
        return soup.get_text("\n", strip=True)

    lines: list[str] = []

    def render(node) -> None:
        if isinstance(node, str):
            text = node.strip()
            if text:
                lines.append(text)
            return

        name = getattr(node, "name", None)
        if name is None:
            for child in node.children:
                render(child)
            return

        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(f"{'#' * level} {text}")
                lines.append("")
            return

        if name == "p":
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(text)
                lines.append("")
            return

        if name in {"ul", "ol"}:
            for idx, li in enumerate(node.find_all("li", recursive=False), start=1):
                prefix = f"{idx}." if name == "ol" else "-"
                text = li.get_text(" ", strip=True)
                if text:
                    lines.append(f"{prefix} {text}")
            lines.append("")
            return

        if name == "pre":
            code = node.get_text("\n", strip=False).strip("\n")
            lang = ""
            code_tag = node.find("code")
            if code_tag and code_tag.get("class"):
                for cls in code_tag["class"]:
                    if cls.startswith("language-"):
                        lang = cls.replace("language-", "")
            lines.append(f"```{lang}")
            lines.append(code)
            lines.append("```")
            lines.append("")
            return

        if name == "blockquote":
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(f"> {text}")
                lines.append("")
            return

        if name == "table":
            rows = []
            for tr in node.find_all("tr"):
                cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                if len(rows) > 1:
                    cols = rows[0].count("|") - 1
                    rows.insert(1, "| " + " | ".join(["---"] * cols) + " |")
                lines.extend(rows)
                lines.append("")
            return

        for child in node.children:
            render(child)

    render(article)
    return "\n".join(lines).strip() + "\n"


def discover_mech_interp_urls(base_url: str) -> list[dict[str, str]]:
    pages: dict[str, dict[str, str]] = {}

    def add(url: str, title: str = "") -> None:
        full = urljoin(base_url, url)
        parsed = urlparse(full)
        if parsed.netloc and parsed.netloc not in {"learnmechinterp.com", "www.learnmechinterp.com"}:
            return
        path = parsed.path.rstrip("/") + "/"
        if path in {"/", "/topics/", "/search/"}:
            return
        if not (path.startswith("/topics/") or path.startswith("/glossary/")):
            return
        if path.count("/") < 3:
            return
        pages[path] = {"url": urljoin(base_url, path), "title": title, "path": path}

    for seed in ("/topics/", "/glossary/", "/"):
        html = fetch(urljoin(base_url, seed)).text
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            title = link.get_text(" ", strip=True)
            add(href, title)

    ordered = sorted(pages.values(), key=lambda item: item["path"])
    return ordered


def scrape_mech_interp_site(*, output_dir: Path, workers: int = 8) -> dict:
    base_url = "https://learnmechinterp.com/"
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages = discover_mech_interp_urls(base_url)
    index_lines = ["# Learn Mechanistic Interpretability", "", "## Pages", ""]
    for page in pages:
        index_lines.append(f"- [{page['title'] or page['path']}]({page['url']})")
    (output_dir / "llms.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    results: list[dict] = []
    failures: list[dict] = []

    def download(page: dict) -> dict:
        response = fetch(page["url"])
        soup = BeautifulSoup(response.text, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else page["title"]
        markdown = html_to_markdown(response.text)
        rel_path = page["path"].strip("/") + ".md"
        local_path = pages_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        header = f"> Source: {page['url']}\n> Title: {title}\n\n"
        local_path.write_text(header + markdown, encoding="utf-8")
        return {
            "title": title,
            "url": page["url"],
            "path": str(local_path.relative_to(output_dir)),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download, page): page for page in pages}
        for future in as_completed(futures):
            page = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failures.append({"url": page["url"], "error": str(exc)})

    manifest = {
        "source": base_url,
        "docs_site": base_url,
        "total_pages": len(pages),
        "successful": len(results),
        "failed": len(failures),
        "format": "markdown-from-html",
        "pages_dir": "pages",
        "index": "llms.txt",
        "failures": failures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[mech_interp] {len(results)}/{len(pages)} pages saved to {output_dir}")
    if failures:
        print(f"[mech_interp] {len(failures)} failures")
    return manifest


def main() -> int:
    targets = sys.argv[1:] or ["crewai", "mech_interp"]
    if "crewai" in targets:
        scrape_llms_site(
            name="crewai",
            index_url="https://docs.crewai.com/llms.txt",
            base_url="https://docs.crewai.com/",
            output_dir=ROOT / "crew_AI_docs",
        )
    if "mech_interp" in targets:
        scrape_mech_interp_site(output_dir=ROOT / "mech_interp_docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
