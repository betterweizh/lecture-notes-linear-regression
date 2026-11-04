"""Number chapters continuously across several numbered toctrees.

Sphinx restarts section numbers at 1 in every ``:numbered:`` toctree. The
book groups chapters into parts, one toctree per part, so without this
extension Part II would start again at "1". After Sphinx assigns numbers,
this extension replaces the chapter number with the chapter's position in
the root document, keeping the lower-level numbers unchanged.
"""

from __future__ import annotations

from docutils import nodes


def _renumber(app, env):
    chapters = env.toctree_includes.get(env.config.root_doc, [])
    for position, docname in enumerate(chapters, start=1):
        numbers = env.toc_secnumbers.get(docname)
        if not numbers:
            continue
        renumbered = {
            anchor: (position,) + tuple(number[1:])
            for anchor, number in numbers.items()
        }
        env.toc_secnumbers[docname] = renumbered
        # The document title node carries its own copy of the number; it is
        # used in the browser tab title and in the previous/next links.
        title = env.titles.get(docname)
        if title is not None and title.get("secnumber"):
            title["secnumber"] = [position] + list(title["secnumber"][1:])
        # Figures, tables, equations and theorem-like blocks are numbered
        # "chapter.n"; replace their chapter part as well.
        for figtype, ids in env.toc_fignumbers.get(docname, {}).items():
            for node_id, number in ids.items():
                if len(number) > 1:
                    ids[node_id] = (position,) + tuple(number[1:])
        toc = env.tocs.get(docname)
        if toc is None:
            continue
        for ref in toc.findall(nodes.reference):
            if "secnumber" in ref:
                ref["secnumber"] = renumbered.get(
                    ref.get("anchorname", ""), ref["secnumber"]
                )
    return []


def setup(app):
    # Run after Sphinx's own TocTreeCollector (default priority 500).
    app.connect("env-get-updated", _renumber, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
