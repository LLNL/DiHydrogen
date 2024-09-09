////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/patterns/factory/CopyFactory.hpp"

#include <memory>
#include <typeinfo>

#include <catch2/catch_test_macros.hpp>

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
  virtual int Data() const noexcept = 0;
  virtual ~WidgetBase() = default;
};

struct Widget : WidgetBase
{
  Widget(int d) : data_(d) {}
  Widget* Copy() const { return new Widget(*this); }

  int Data() const noexcept override { return data_; }
  int data_;
};

struct Gizmo : WidgetBase
{
  Gizmo(int d) : data_(d) {}
  std::unique_ptr<Gizmo> Clone() const
  {
    return h2::ToUnique(new Gizmo(*this));
  }

  int Data() const noexcept override { return data_; }
  int data_;
};

std::unique_ptr<WidgetBase> CopyGizmo(WidgetBase const& obj)
{
  auto const& gizmo = dynamic_cast<Gizmo const&>(obj);
  return gizmo.Clone();
}

std::unique_ptr<WidgetBase> CopyWidget(WidgetBase const& obj)
{
  auto const& widget = dynamic_cast<Widget const&>(obj);
  return h2::ToUnique(widget.Copy());
}

} // namespace

TEST_CASE("testing the copy factory class", "[factory][utilities]")
{
  using WidgetFactory = h2::factory::CopyFactory<WidgetBase>;

  WidgetFactory factory;
  SECTION("Register new classes")
  {
    CHECK(factory.register_builder(typeid(Gizmo), CopyGizmo));
    CHECK(factory.register_builder(typeid(Widget), CopyWidget));
    CHECK(factory.size() == 2UL);

    SECTION("Re-registering a type fails.")
    {
      CHECK_FALSE(factory.register_builder(typeid(Widget), CopyWidget));
      CHECK(factory.size() == 2UL);
    }

    SECTION("Copy objects by concrete type")
    {
      Widget w(17);
      auto w2 = factory.copy_object(w);
      auto const& w2_ref = *w2;

      CHECK(w2->Data() == w.Data());
      CHECK(typeid(w2_ref) == typeid(w));

      Gizmo g(13);
      auto g2 = factory.copy_object(g);
      auto const& g2_ref = *g2;

      CHECK(g2->Data() == g.Data());
      CHECK(typeid(g2_ref) == typeid(g));
    }

    SECTION("Copy objects through base type")
    {
      auto g = std::unique_ptr<WidgetBase>(new Gizmo(37));
      auto w = std::unique_ptr<WidgetBase>(new Widget(73));

      auto w2 = factory.copy_object(*w);
      auto g2 = factory.copy_object(*g);

      CHECK(w2->Data() == w->Data());
      CHECK(g2->Data() == g->Data());
    }

    SECTION("Get list of supported types")
    {
      auto names = factory.registered_types();

      CHECK(names.size() == factory.size());
    }

    SECTION("Unregister types.")
    {
      CHECK(factory.unregister(typeid(Widget)));
      CHECK(factory.size() == 1UL);

      CHECK(factory.unregister(typeid(Gizmo)));
      CHECK(factory.size() == 0UL);
    }
  }
  SECTION("Cannot copy unregistered widgets")
  {
    Gizmo g(13);
    CHECK_THROWS(factory.copy_object(g));
  }
}
