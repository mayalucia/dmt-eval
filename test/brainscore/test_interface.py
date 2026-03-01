"""Tests for the Interface base class."""

import pytest
from brain_score_dmt.interface import Interface


# --- Test Interfaces ---

class Arithmetic(Interface):
    def add(self, x, y): ...
    def subtract(self, x, y): ...


class Greetable(Interface):
    def greet(self, name): ...


# --- Test Implementations ---

class GoodCalculator:
    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y


class PartialCalculator:
    """Has add but not subtract."""
    def add(self, x, y):
        return x + y


class EmptyClass:
    pass


class Greeter:
    def greet(self, name):
        return f"Hello, {name}"


# --- Tests ---

class TestInterfaceDeclaration:
    def test_required_methods_captured(self):
        assert Arithmetic.__required_methods__ == {'add', 'subtract'}

    def test_single_method_interface(self):
        assert Greetable.__required_methods__ == {'greet'}

    def test_each_interface_has_own_registry(self):
        assert Arithmetic.__implementation_registry__ is not Greetable.__implementation_registry__

    def test_base_interface_has_empty_requirements(self):
        assert Interface.__required_methods__ == set()


class TestValidation:
    def test_valid_implementation(self):
        assert Arithmetic.validate(GoodCalculator) == []

    def test_missing_one_method(self):
        missing = Arithmetic.validate(PartialCalculator)
        assert missing == ['subtract']

    def test_missing_all_methods(self):
        missing = Arithmetic.validate(EmptyClass)
        assert set(missing) == {'add', 'subtract'}

    def test_is_implemented_by_true(self):
        assert Arithmetic.is_implemented_by(GoodCalculator)

    def test_is_implemented_by_false(self):
        assert not Arithmetic.is_implemented_by(PartialCalculator)

    def test_validates_instances_too(self):
        """validate() works on instances, not just classes."""
        assert Arithmetic.validate(GoodCalculator()) == []

    def test_cross_interface_no_confusion(self):
        """Greeter implements Greetable but not Arithmetic."""
        assert Greetable.is_implemented_by(Greeter)
        assert not Arithmetic.is_implemented_by(Greeter)


class TestRegistration:
    def test_register_valid(self):
        Arithmetic.register('good', GoodCalculator)
        assert 'good' in Arithmetic.implementations()
        assert Arithmetic.implementations()['good'] is GoodCalculator

    def test_register_invalid_raises(self):
        with pytest.raises(TypeError, match="Missing.*subtract"):
            Arithmetic.register('partial', PartialCalculator)

    def test_register_does_not_cross_interfaces(self):
        Greetable.register('greeter', Greeter)
        assert 'greeter' in Greetable.implementations()
        assert 'greeter' not in Arithmetic.implementations()


class TestImplementationGuide:
    def test_guide_lists_methods(self):
        guide = Arithmetic.implementation_guide()
        assert 'add' in guide
        assert 'subtract' in guide
        assert 'Arithmetic' in guide


class TestPropertyInterface:
    """Properties should be captured as required methods."""

    def test_property_in_interface(self):
        class HasIdentifier(Interface):
            @property
            def identifier(self): ...

        assert 'identifier' in HasIdentifier.__required_methods__

    def test_property_validated(self):
        class HasIdentifier(Interface):
            @property
            def identifier(self): ...

        class WithId:
            @property
            def identifier(self):
                return 'test'

        class WithoutId:
            pass

        assert HasIdentifier.is_implemented_by(WithId)
        assert not HasIdentifier.is_implemented_by(WithoutId)
