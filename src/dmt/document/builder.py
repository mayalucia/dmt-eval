# SPDX-License-Identifier: MIT
# dmt-eval — https://github.com/mayalucia/dmt-eval
#
# Original architecture: Blue Brain Project / EPFL (2017-2024)
# Modernised for dmt-eval by Vishal Sood + AI collaborators, 2026.

"""
Build a document using decorators.

A ``DocumentBuilder`` accumulates section definitions via decorator syntax,
providing a concise alternative to instantiating ``Document`` / ``LabReport``
objects with deeply nested keyword arguments.

Example
-------
::

    document = DocumentBuilder("Article")

    @document.abstract
    def _():
        \"\"\"We analyze the densities of cortical layers.\"\"\"
        pass

    @document.section("Introduction")
    def _():
        \"\"\"The cortex is organized into layers...\"\"\"
        pass

    @document.section("Introduction").illustration
    def scaffold():
        \"\"\"A digitally reconstructed neocortical column.\"\"\"
        return Path("resources/neocortical_scaffold.png")
"""

import re
from collections import OrderedDict


_PUNCT = re.compile(r"[,:&#/\\$?^;.]")


def make_label(string, separator="_"):
    """Turn a human string into a snake_case label."""
    if not isinstance(string, str):
        string = str(string)
    words = _PUNCT.sub("", string.strip()).lower().split()
    return separator.join(w for w in words if w)


__all__ = ["DocumentBuilder", "SectionProxy", "make_label"]


# ── Section names that LabReport knows about ──────────────────────────────────

_KNOWN_SECTIONS = frozenset({
    "abstract", "introduction", "methods",
    "results", "discussion", "conclusion",
})

# ── SectionProxy ──────────────────────────────────────────────────────────────

class SectionProxy:
    """Accumulator for a single document section.

    Returned by ``DocumentBuilder.section(name)`` and by attribute access
    on known section names (``document.introduction``).  Callable as a
    decorator to register the section's primary function (whose docstring
    becomes the narrative text).  Sub-elements are registered via chained
    attribute decorators::

        @document.section("Introduction").illustration
        def scaffold():
            ...
    """

    def __init__(self, name):
        self.name = name
        self.label = make_label(name)
        self._functions = OrderedDict()

    # ── primary decorator ─────────────────────────────────────────────────

    def __call__(self, fn):
        """Register *fn* under its own name.

        If the section already has a function with the same name, it is
        replaced.  This allows calling ``@document.section("X")`` twice
        with differently-named functions to register e.g. both a
        measurement and an illustration.
        """
        key = fn.__name__
        # Use the role-specific key when it matches a known sub-element,
        # otherwise store under the function's own name.
        self._functions[key] = fn
        return fn

    # ── sub-element decorators ────────────────────────────────────────────

    @property
    def illustration(self):
        """Decorator that registers an illustration function."""
        return _SubElementDecorator(self, "illustration")

    @property
    def tables(self):
        """Decorator that registers a tables / data function."""
        return _SubElementDecorator(self, "tables")

    @property
    def parameters(self):
        """Decorator that registers a parameters function."""
        return _SubElementDecorator(self, "parameters")

    @property
    def measurement(self):
        """Decorator that registers a measurement function."""
        return _SubElementDecorator(self, "measurement")

    # ── introspection ─────────────────────────────────────────────────────

    def get_narrative(self):
        """Return the narrative text collected from function docstrings.

        The first registered function whose name is ``_`` (anonymous) or
        that doesn't match a known sub-element key provides the narrative
        via its docstring.
        """
        sub_keys = {"illustration", "tables", "parameters", "measurement"}
        for key, fn in self._functions.items():
            if key not in sub_keys:
                doc = getattr(fn, "__doc__", None)
                if doc:
                    return doc.strip()
        return None

    def get_function(self, role):
        """Return the function registered under *role*, or ``None``."""
        return self._functions.get(role)

    def __repr__(self):
        roles = list(self._functions.keys())
        return f"SectionProxy({self.name!r}, functions={roles})"

class _SubElementDecorator:
    """A callable that, when used as a decorator, stores *fn* in the
    parent ``SectionProxy`` under the given *role* key."""

    def __init__(self, section_proxy, role):
        self._section = section_proxy
        self._role = role

    def __call__(self, fn):
        self._section._functions[self._role] = fn
        return fn

    def __repr__(self):
        return (f"_SubElementDecorator("
                f"{self._section.name!r}, {self._role!r})")

# ── DocumentBuilder ───────────────────────────────────────────────────────────

class DocumentBuilder:
    """Accumulate section definitions via decorators, then optionally
    build a ``Document`` or ``LabReport`` instance.

    Parameters
    ----------
    document_class_or_title : str or type
        Either a plain string title (produces a generic document) or a
        ``Document`` subclass such as ``LabReport``.
    **kwargs
        Forwarded to the document class constructor when ``build()`` is
        called.  When *document_class_or_title* is a string, ``title``
        is set automatically.
    """

    def __init__(self, document_class_or_title, **kwargs):
        if isinstance(document_class_or_title, str):
            self._document_class = None
            self._title = document_class_or_title
        else:
            self._document_class = document_class_or_title
            self._title = kwargs.pop("title", None)
        self._kwargs = kwargs
        self._sections = OrderedDict()
        self._interface_methods = []

    # ── named section access ──────────────────────────────────────────────

    def section(self, name):
        """Return (or create) the ``SectionProxy`` for *name*.

        Calling ``document.section("X")`` twice returns the *same* proxy,
        so decorators accumulate on a single section.
        """
        label = make_label(name)
        if label not in self._sections:
            self._sections[label] = SectionProxy(name)
        return self._sections[label]

    # ── known-section shortcuts via __getattr__ ───────────────────────────

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _KNOWN_SECTIONS:
            return self.section(name.replace("_", " ").title()
                                if "_" in name else name.capitalize()
                                if name == name.lower() else name)
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}")

    # ── interfacemethod decorator ─────────────────────────────────────────

    @property
    def interfacemethod(self):
        """Decorator that registers a function as an interface method."""
        return _InterfaceMethodDecorator(self)

    # ── introspection ─────────────────────────────────────────────────────

    @property
    def title(self):
        return self._title

    @property
    def sections(self):
        """Return an ``OrderedDict`` of label -> ``SectionProxy``."""
        return OrderedDict(self._sections)

    @property
    def interface_methods(self):
        return list(self._interface_methods)

    def __repr__(self):
        cls = (self._document_class.__name__
               if self._document_class else "Document")
        sections = list(self._sections.keys())
        return f"DocumentBuilder({cls}, title={self._title!r}, sections={sections})"


class _InterfaceMethodDecorator:
    """Callable that stores the decorated function in the builder's
    interface method list."""

    def __init__(self, builder):
        self._builder = builder

    def __call__(self, fn):
        self._builder._interface_methods.append(fn)
        return fn
