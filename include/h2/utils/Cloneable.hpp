#pragma once

#include <memory>

namespace h2
{

/** @file
 *
 *  This file implements covariant returns via smart pointers for a
 *  polymorphic @c clone function. The implementation largely follows
 *  the solution presented <a
 *  href="https://www.fluentcpp.com/2017/09/12/how-to-return-a-smart-pointer-and-use-covariance/">by
 *  the FluentC++ blog</a>. Some class/tag names have been updated to
 *  be clearer, in my opinion.
 *
 *  This code was (mostly) copied from LBANN; it will be removed from
 *  LBANN soon, and LBANN will be updated to use this instead.
 *
 *  This file requires C++17.
 */

/** @brief Declare @c Base to be a virtual base.
 *
 *  This metafunction adds @c Base as a virtual base
 *  class. Constructors of @c Base are added to this class.
 *
 *  @tparam Base The class to be declared as a virtual base.
 */
template <typename Base>
struct AsVirtual : virtual Base
{
    using Base::Base;
};

/** @brief Declare that @c T has unimplemented virtual functions.
 *
 *  Due to metaprogramming restrictions on CRTP interfaces (namely,
 *  the requirement on the input type to std::is_abstract must be a
 *  complete class), we rely on the user of these mechanisms to
 *  declare when a class has unimplemented virtual functions (or "is
 *  abstract").
 *
 *  @tparam T The type that has at least one unimplemented virtual
 *  function.
 */
template <typename T>
struct Abstract
{};

/** @brief Inject polymorphic clone functions into hierarchies.
 *
 *  This class uses CRTP to inject the derived class's clone()
 *  function directly into the class and uses
 *  <a href="http://www.gotw.ca/publications/mill18.htm">the
 *  Template Method</a> to virtualize it.
 *
 *  @tparam T The concrete class to be cloned.
 *  @tparam Base The base class(es) of T.
 */
template <typename T, typename... Base>
class Cloneable : public Base...
{
public:
    /** @brief Return an exception-safe, memory-safe copy of this object. */
    std::unique_ptr<T> clone() const
    {
        return std::unique_ptr<T>{static_cast<T*>(this->do_clone_())};
    }

protected:
    using Base::Base...;

private:
    /** @brief Implement the covariant raw-pointer-based clone operation. */
    virtual Cloneable* do_clone_() const override
    {
        return new T{static_cast<T const&>(*this)};
    }
}; // class Cloneable

/** @brief Specialization of Cloneable to handle stand-alone classes. */
template <typename T>
class Cloneable<T>
{
public:
    virtual ~Cloneable() = default;

    std::unique_ptr<T> clone() const
    {
        return std::unique_ptr<T>{static_cast<T*>(this->do_clone_())};
    }

private:
    virtual Cloneable* do_clone_() const
    {
        return new T{static_cast<T const&>(*this)};
    }
}; // class Cloneable<T>

/** @brief Specialization of Cloneable for intermediate classes.
 *
 *  Classes that are neither the top of the hierarchy nor a leaf of
 *  the class tree should be virtual. An unfortunate consequence of
 *  the CRTP method is that the target of the CRTP, @c T in this case,
 *  is not a complete class when this class is instantiated, so
 *  metaprogramming based on @c T is very restricted. Thus, users must
 *  tag the target class with Abstract. Doing so will
 *  ensure that the @c do_clone_() function is declared pure virtual.
 */
template <typename T, typename... Base>
class Cloneable<Abstract<T>, Base...> : public Base...
{
public:
    std::unique_ptr<T> clone() const
    {
        return std::unique_ptr<T>{static_cast<T*>(this->do_clone_())};
    }

protected:
    using Base::Base...;

private:
    virtual Cloneable* do_clone_() const = 0;
};

/** @brief Specialization of Cloneable to handle the top of hierarchies. */
template <typename T>
class Cloneable<Abstract<T>>
{
public:
    virtual ~Cloneable() = default;

    std::unique_ptr<T> clone() const
    {
        return std::unique_ptr<T>{static_cast<T*>(this->do_clone_())};
    }

private:
    virtual Cloneable* do_clone_() const = 0;

}; // class Cloneable<T>

} // namespace h2
