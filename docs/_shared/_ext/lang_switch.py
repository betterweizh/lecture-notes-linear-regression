"""English/Chinese language switch.

Both editions use identical file names, so the counterpart of
``en/04-anova-and-inference.html`` is ``zh/04-anova-and-inference.html``.
For every page this extension computes the relative URL of the same page in
each other edition and exposes it to the ``lang-switch.html`` template.

Expected output layout::

    _build/html/
        index.html      (redirects to the reader's preferred language)
        en/...
        zh/...
"""

from __future__ import annotations


def _add_context(app, pagename, templatename, context, doctree):
    current = app.config.lang_switch_current
    languages = app.config.lang_switch_languages or {}
    depth = pagename.count("/")
    up = "../" * (depth + 1)
    suffix = app.builder.out_suffix  # ".html"
    page = "index" if pagename == "genindex" else pagename
    alternates = []
    for code, info in languages.items():
        if code == current:
            continue
        alternates.append(
            {
                "code": code,
                "label": info["label"],
                "short": info["short"],
                "url": f"{up}{code}/{page}{suffix}",
            }
        )
    context["lang_switch_current"] = current
    context["lang_switch_current_label"] = languages.get(current, {}).get("label", current)
    context["lang_switch_alternates"] = alternates


def setup(app):
    app.add_config_value("lang_switch_current", "en", "html")
    app.add_config_value("lang_switch_languages", {}, "html")
    # Run before pydata-sphinx-theme removes templates that render empty.
    app.connect("html-page-context", _add_context, priority=100)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
