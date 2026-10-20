"""Help-center HTML: keep the article, drop the page.

Boilerplate (navigation, cookie banner, footer, "related articles", feedback widget) repeats on every
page and would make every page match "fees" or "wire". Hidden elements and comments are dropped too:
on a CMS they hold stale A/B variants and editor notes, which must never be cited.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from ..corpus import Block
from .tables import row_block

BOILERPLATE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "svg",
    "iframe",
)
BOILERPLATE_SELECTORS = (
    "[hidden]",
    "[aria-hidden=true]",
    "[role=dialog]",
    # Help-center template parts without a semantic tag (site-specific extraction rules).
    ".cookie-banner",
    ".article-feedback",
    ".article-meta",
)
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
BLOCK_TAGS = {
    "p",
    "ul",
    "ol",
    "table",
    "details",
    "dl",
    "blockquote",
    "pre",
    "div",
    "section",
    "article",
    "main",
    *HEADINGS,
}


def _text(node: Tag) -> str:
    # No separator: inline tags must not split "Lock card</strong>." into "Lock card ."; the source's own
    # whitespace (and <br>, replaced by a space in _strip_boilerplate) separates words.
    return " ".join(node.get_text().split())


def _strip_boilerplate(soup: BeautifulSoup) -> None:
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()
    for tag in soup.select(", ".join(BOILERPLATE_SELECTORS)):
        tag.decompose()
    for tag in soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", "").lower()):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with(" ")


class _Walker:
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.path: list[tuple[int, str]] = []

    def section(self) -> tuple[str, ...]:
        return tuple(text for _, text in self.path)

    def emit(self, kind: str, text: str, node: Tag) -> None:
        line = node.sourceline or 0
        if text:
            self.blocks.append(Block(kind, text, self.section(), line, line))

    def heading(self, level: int, text: str, node: Tag) -> None:
        if level == 1:  # the page title is already the document title
            self.path.clear()
            return
        while self.path and self.path[-1][0] >= level:
            self.path.pop()
        self.path.append((level, text))
        self.emit("heading", text, node)

    def walk(self, node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if child.strip() and isinstance(node, Tag) and node.name in {"div", "section", "article", "main"}:
                    self.blocks.append(
                        Block(
                            "text", " ".join(child.split()), self.section(), node.sourceline or 0, node.sourceline or 0
                        )
                    )
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in HEADINGS:
                self.heading(HEADINGS[name], _text(child), child)
            elif name == "p" or name in {"blockquote", "pre"}:
                self.emit("text", _text(child), child)
            elif name in {"ul", "ol"}:
                for number, item in enumerate(child.find_all("li", recursive=False), 1):
                    nested = item.find_all(["ul", "ol"])
                    for sub in nested:
                        sub.extract()
                    prefix = f"{number}. " if name == "ol" else ""
                    self.emit("text", prefix + _text(item), item)
                    for sub in nested:
                        self.walk(BeautifulSoup(str(sub), "html.parser"))
            elif name == "table":
                self.table(child)
            elif name == "details":
                summary = child.find("summary")
                question = _text(summary) if summary else ""
                if summary:
                    summary.extract()
                depth = (self.path[-1][0] if self.path else 1) + 1
                saved = list(self.path)
                self.heading(depth, question, child)
                self.walk(child)
                self.path = saved
            elif name == "dl":
                for term in child.find_all("dt"):
                    definition = term.find_next_sibling("dd")
                    self.emit("text", f"{_text(term)}: {_text(definition) if definition else ''}".strip(": "), term)
            elif name in BLOCK_TAGS or child.find(list(BLOCK_TAGS)):
                self.walk(child)
            else:
                self.emit("text", _text(child), child)

    def table(self, table: Tag) -> None:
        rows = table.find_all("tr")
        if not rows:
            return
        caption = table.find("caption")
        headers = [_text(c) for c in rows[0].find_all(["th", "td"])]
        section = self.section() + ((_text(caption),) if caption else ())
        for tr in rows[1:]:
            values = [_text(c) for c in tr.find_all(["th", "td"])]
            line = tr.sourceline or 0
            if block := row_block(headers, values, section, (line, line)):
                self.blocks.append(block)


def parse_html(raw: str) -> list[Block]:
    soup = BeautifulSoup(raw, "html.parser")
    _strip_boilerplate(soup)
    # Whitelist the article (the help-center template guarantees <main><article>); fall back gracefully.
    root = soup.select_one("main article") or soup.find("main") or soup.body or soup
    walker = _Walker()
    walker.walk(root)
    return walker.blocks
