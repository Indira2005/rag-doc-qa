import ast
import asyncio
import re
from pathlib import Path

from rag.generate import clean_citations


def clean(parts):
    async def source():
        for part in parts:
            yield part

    async def collect():
        return [t async for t in clean_citations(source())]

    return asyncio.run(collect())


def test_citations_split_across_tokens():
    out = clean(["Heads ", "【", "3", "†L1", "-L3】", " and 【5】【1, 2】. [4]"])
    assert "".join(out) == "Heads [3] and [5][1, 2]. [4]"


def test_normal_tokens_are_not_delayed():
    assert clean(["The", " model"]) == ["The", " model"]


def test_unfinished_marker_is_not_lost():
    assert "".join(clean(["text 【3"])) == "text 【3"


def load(name):
    # load one function from ui.py, without starting Streamlit
    tree = ast.parse((Path(__file__).parent.parent / "ui.py").read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"re": re}
    exec(compile(ast.Module([node], []), "ui", "exec"), scope)
    return scope[name]


def test_citations_show_file_and_page():
    sources = [{"n": 1, "source": "a.pdf", "page": 8}, {"n": 2, "source": "a.pdf", "page": 5},
               {"n": 3, "source": "a.pdf", "page": 5}]
    text, cited = load("add_pages")("Eight [2]. Same [2, 3]. Unknown [9].", sources)
    assert text == "Eight [a.pdf, p. 5]. Same [a.pdf, p. 5]. Unknown [9]."
    assert [s["n"] for s in cited] == [2, 3]


def test_maths_delimiters():
    show = load("show_math")
    assert show(r"size \(d_{k}\) here") == "size $d_{k}$ here"
    assert show("a \\[\nx^2\n\\] [2].") == "a \n\n$$\nx^2\n$$\n\n [2]."
