////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include <h2/patterns/factory/PrototypeFactory.hpp>

#include <memory>
#include <typeinfo>

namespace h2
{
template <typename T>
std::unique_ptr<T> ToUnique(T* ptr)
{
    return std::unique_ptr<T>(ptr);
}
} // namespace h2

namespace
{
struct WidgetBase
{
    virtual WidgetBase* Copy() const = 0;
    virtual int Data() const noexcept = 0;
    virtual ~WidgetBase() = default;
};

struct Widget : WidgetBase
{
    Widget(int d) : data_(d) {}
    Widget* Copy() const override { return new Widget(*this); }

    int Data() const noexcept override { return data_; }
    int data_;
};

struct Gizmo : WidgetBase
{
    Gizmo(int d) : data_(d) {}
    Gizmo* Copy() const override { return new Gizmo(*this); }

    int Data() const noexcept override { return data_; }
    int data_;
};

struct BasicCopyPolicy
{
    std::unique_ptr<WidgetBase> Copy(WidgetBase const& obj) const
    {
        return h2::ToUnique(obj.Copy());
    }
};

} // namespace

TEST_CASE("testing the prototype factory class", "[factory][utilities]")
{
    using WidgetFactory =
        h2::factory::PrototypeFactory<WidgetBase, std::string, BasicCopyPolicy>;

    WidgetFactory factory;
    SECTION("Register new prototypes")
    {
        CHECK(factory.register_prototype("gizmo", h2::ToUnique(new Gizmo(71))));
        CHECK(
            factory.register_prototype("widget", h2::ToUnique(new Widget(17))));
        CHECK(factory.size() == 2UL);

        SECTION("Re-registering a type fails.")
        {
            CHECK_FALSE(factory.register_prototype(
                "widget", h2::ToUnique(new Widget(13))));
            CHECK(factory.size() == 2UL);
        }

        SECTION("Copy objects by their key")
        {
            auto w = factory.copy_prototype("widget");
            auto const& w_ref = *w;

            CHECK(w->Data() == 17);
            CHECK(typeid(w_ref) == typeid(Widget));

            auto g = factory.copy_prototype("gizmo");
            auto const& g_ref = *g;

            CHECK(g->Data() == 71);
            CHECK(typeid(g_ref) == typeid(Gizmo));

            CHECK_THROWS(factory.copy_prototype("invalid key"));
        }

        SECTION("Get list of supported types")
        {
            auto names = factory.registered_ids();

            CHECK(names.size() == factory.size());
        }

        SECTION("Unregister types.")
        {
            CHECK(factory.unregister("widget"));
            CHECK(factory.size() == 1UL);

            CHECK(factory.unregister("gizmo"));
            CHECK(factory.size() == 0UL);

            CHECK_FALSE(factory.unregister("invalid key"));
            CHECK(factory.size() == 0UL);
        }
    }
}
