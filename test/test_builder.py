"""Tests for the DocumentBuilder and SectionProxy."""

from collections import OrderedDict
from dmt.document.builder import DocumentBuilder, SectionProxy


class TestSectionProxy:
    """SectionProxy accumulates decorated functions."""

    def test_callable_as_decorator(self):
        proxy = SectionProxy("Introduction")

        @proxy
        def narrative():
            """The cortex is organized into layers."""
            pass

        assert "narrative" in proxy._functions
        assert proxy._functions["narrative"] is narrative

    def test_illustration_sub_decorator(self):
        proxy = SectionProxy("Introduction")

        @proxy.illustration
        def scaffold():
            """A cortical column."""
            return "scaffold.png"

        assert "illustration" in proxy._functions
        assert proxy._functions["illustration"] is scaffold

    def test_tables_sub_decorator(self):
        proxy = SectionProxy("Methods")

        @proxy.tables
        def reference_data():
            """Experimental reference."""
            return {"cell_density": "data.csv"}

        assert "tables" in proxy._functions

    def test_parameters_sub_decorator(self):
        proxy = SectionProxy("Methods")

        @proxy.parameters
        def layer_params(adapter, model):
            return adapter.get_layers(model)

        assert "parameters" in proxy._functions

    def test_get_narrative_from_docstring(self):
        proxy = SectionProxy("Abstract")

        @proxy
        def _():
            """We analyze cell densities across cortical layers."""
            pass

        assert proxy.get_narrative() == "We analyze cell densities across cortical layers."

    def test_get_narrative_skips_sub_elements(self):
        proxy = SectionProxy("Introduction")

        @proxy
        def _():
            """The cortex has layers."""
            pass

        @proxy.illustration
        def img():
            """Caption for the image."""
            return "img.png"

        assert proxy.get_narrative() == "The cortex has layers."

    def test_same_section_accumulates(self):
        proxy = SectionProxy("cell_density")

        @proxy
        def measurement():
            """Measure cell density."""
            pass

        @proxy
        def illustration():
            """Plot cell density."""
            pass

        assert "measurement" in proxy._functions
        assert "illustration" in proxy._functions


class TestDocumentBuilder:
    """DocumentBuilder routes decorators to SectionProxy instances."""

    def test_string_title(self):
        doc = DocumentBuilder("Article")
        assert doc.title == "Article"

    def test_class_with_title(self):
        class FakeReport:
            pass
        doc = DocumentBuilder(FakeReport, title="Analysis")
        assert doc.title == "Analysis"
        assert doc._document_class is FakeReport

    def test_section_creates_proxy(self):
        doc = DocumentBuilder("Article")
        proxy = doc.section("Introduction")
        assert isinstance(proxy, SectionProxy)
        assert proxy.name == "Introduction"

    def test_section_returns_same_proxy(self):
        doc = DocumentBuilder("Article")
        p1 = doc.section("Introduction")
        p2 = doc.section("Introduction")
        assert p1 is p2

    def test_known_section_via_getattr(self):
        doc = DocumentBuilder("Article")
        proxy = doc.abstract
        assert isinstance(proxy, SectionProxy)
        assert proxy.label == "abstract"

    def test_abstract_decorator(self):
        doc = DocumentBuilder("Article")

        @doc.abstract
        def _():
            """We analyze cortical layers."""
            pass

        assert "abstract" in doc.sections
        assert doc.sections["abstract"].get_narrative() == "We analyze cortical layers."

    def test_chained_sub_element(self):
        doc = DocumentBuilder("Article")

        @doc.section("Introduction").illustration
        def scaffold():
            """A cortical column."""
            return "scaffold.png"

        proxy = doc.sections["introduction"]
        assert proxy.get_function("illustration") is scaffold

    def test_interfacemethod(self):
        doc = DocumentBuilder("Article")

        @doc.interfacemethod
        def get_layers(adapter, model):
            pass

        assert len(doc.interface_methods) == 1
        assert doc.interface_methods[0] is get_layers

    def test_full_workflow(self):
        """Exercise the complete decorator workflow from test_section_builder."""
        doc = DocumentBuilder("Article")

        @doc.abstract
        def _():
            """We analyze the densities of cortical layers."""
            pass

        @doc.section("Introduction")
        def _():
            """Cortical area is composed of layers of cells."""
            pass

        @doc.section("Introduction").illustration
        def neocortical_scaffold():
            """The neocortex is a 2-3 mm thick sheet of tissue."""
            return "resources/neocortical_scaffold.png"

        @doc.section("Introduction").tables
        def experimental_cell_density():
            """Mock experimental cell density."""
            return {"data": "resources/experimental_cell_density.csv"}

        @doc.section("cell_density")
        def measurement():
            """Layer cell densities for regions."""
            pass

        @doc.section("cell_density")
        def illustration():
            """Mock cell density plot."""
            pass

        assert len(doc.sections) == 3
        assert "abstract" in doc.sections
        assert "introduction" in doc.sections
        assert "cell_density" in doc.sections

        intro = doc.sections["introduction"]
        assert intro.get_narrative() == "Cortical area is composed of layers of cells."
        assert intro.get_function("illustration") is neocortical_scaffold
        assert intro.get_function("tables") is experimental_cell_density

        cd = doc.sections["cell_density"]
        assert "measurement" in cd._functions
        assert "illustration" in cd._functions
