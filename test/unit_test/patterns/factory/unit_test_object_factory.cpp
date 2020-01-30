////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include <h2/patterns/factory/ObjectFactory.hpp>

#include <typeinfo>

namespace
{
struct WidgetBase
{
    virtual ~WidgetBase() = default;
};
struct Widget : WidgetBase
{};
struct Gizmo : WidgetBase
{};

std::unique_ptr<WidgetBase> MakeWidget()
{
    return std::unique_ptr<Widget>(new Widget);
}

std::unique_ptr<WidgetBase> MakeGizmo()
{
    return std::unique_ptr<Gizmo>(new Gizmo);
}

std::unique_ptr<WidgetBase> MakeGizmo2()
{
    return std::unique_ptr<Gizmo>(new Gizmo);
}

// What follows is an abstraction to allow testing multiple types of
// keys. Currently tested types are string and int. This should be
// sufficient for establishing that the invariants hold.

enum class GenericKey
{
    INVALID,
    WIDGET,
    GIZMO
};

template <typename T>
struct Key;

template <>
struct Key<std::string>
{
    static std::string get(GenericKey key)
    {
        switch (key)
        {
        case GenericKey::WIDGET: return "Widget";
        case GenericKey::GIZMO: return "Gizmo";
        case GenericKey::INVALID: return "Invalid";
        }
    }
};

template <>
struct Key<int>
{
    static int get(GenericKey key) noexcept { return static_cast<int>(key); }
};

} // namespace

TEMPLATE_TEST_CASE(
    "testing the factory class", "[factory][utilities]", std::string, int)
{
    using WidgetFactory = h2::factory::ObjectFactory<WidgetBase, TestType>;
    using key = Key<TestType>;

    WidgetFactory factory;

    SECTION("New builders are registered")
    {
        CHECK(
            factory.register_builder(key::get(GenericKey::WIDGET), MakeWidget));

        CHECK(factory.register_builder(key::get(GenericKey::GIZMO), MakeGizmo));

        CHECK(factory.size() == 2UL);

        // Verify the keys
        auto names = factory.registered_ids();
        for (auto const& name : names)
            CHECK(
                (name == key::get(GenericKey::WIDGET)
                 || name == key::get(GenericKey::GIZMO)));
    }

    SECTION("A key is used multiple times")
    {
        CHECK(factory.register_builder(key::get(GenericKey::GIZMO), MakeGizmo));

        CHECK_FALSE(
            factory.register_builder(key::get(GenericKey::GIZMO), MakeGizmo2));

        CHECK(factory.size() == 1UL);
    }

    SECTION("A new object is requested with a valid key")
    {
        CHECK(
            factory.register_builder(key::get(GenericKey::WIDGET), MakeWidget));

        CHECK(factory.register_builder(key::get(GenericKey::GIZMO), MakeGizmo));

        std::unique_ptr<WidgetBase> g, w;

        REQUIRE_NOTHROW(
            w = factory.create_object(key::get(GenericKey::WIDGET)));
        auto const& w_ref = *w;
        CHECK(typeid(w_ref) == typeid(Widget));

        REQUIRE_NOTHROW(g = factory.create_object(key::get(GenericKey::GIZMO)));
        auto const& g_ref = *g;
        CHECK(typeid(g_ref) == typeid(Gizmo));
    }

    SECTION("A new object is requested with with an invalid key")
    {
        CHECK_THROWS_WITH(
            factory.create_object(key::get(GenericKey::INVALID)),
            "Unknown type identifier.");
    }

    SECTION("Keys are removed")
    {
        CHECK(
            factory.register_builder(key::get(GenericKey::WIDGET), MakeWidget));
        CHECK(
            factory.register_builder(key::get(GenericKey::GIZMO), MakeGizmo2));

        CHECK(factory.unregister(key::get(GenericKey::WIDGET)));
        CHECK(factory.size() == 1UL);

        SECTION("The remaining key is correct.")
        {
            auto names = factory.registered_ids();
            CHECK(std::distance(names.begin(), names.end()) == 1UL);
            CHECK(names.front() == key::get(GenericKey::GIZMO));
        }

        SECTION("The remaining builder still works.")
        {
            std::unique_ptr<WidgetBase> g;
            CHECK_NOTHROW(
                g = factory.create_object(key::get(GenericKey::GIZMO)));
            auto const& g_ref = *g;
            CHECK(typeid(g_ref) == typeid(Gizmo));
        }

        SECTION("The removed key is invalid.")
        {
            std::unique_ptr<WidgetBase> obj;
            CHECK_THROWS_AS(
                obj = factory.create_object(key::get(GenericKey::WIDGET)),
                std::exception);
        }
    }
}
