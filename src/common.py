"""Small helpers shared by every retriever."""


def doc_text(doc) -> str:
    """Return the searchable text of a corpus entry.

    Corpus entries may be a plain string or a dict with 'title' and 'text'.
    We join title and text because a SciFact title is part of the evidence.
    """
    if isinstance(doc, str):
        return doc
    title = doc.get("title") or ""
    text = doc.get("text") or ""
    return f"{title}. {text}".strip(". ").strip()