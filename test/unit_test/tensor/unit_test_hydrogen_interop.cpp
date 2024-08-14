////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/hydrogen_interop.hpp"
#include "utils.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{

template <hydrogen::Device D>
El::Matrix<DataType, D>
make_matrix(El::Int height, El::Int width, El::Int ldim = 0)
{
    using MatrixType = El::Matrix<DataType, D>;
    MatrixType mat{height, width, ldim};
    El::MakeUniform(mat);
    return mat;
}

} // namespace

TEST_CASE("is_chw_packed predicate", "[tensor][utilities]")
{
    // This is all metadata, so just test the CPU type
    constexpr h2::Device Dev = h2::Device::CPU;
    using TensorType = h2::Tensor<DataType>;

    // This is simply a pointer that can be used to construct a tensor
    // using the "pointer-attach" constructors. The data is never read
    // in these tests, which just exercise metadata capabilities, so
    // it doesn't matter that the pointer is invalid.
    // Don't use nullptr because that is checked.
    DataType* mock_data = reinterpret_cast<DataType*>(1);

    SECTION("Empty tensor is considered trivially packed.")
    {
        TensorType tensor{Dev};
        CHECK(is_chw_packed(tensor));
    }

    SECTION("Rank one, unit stride tensor")
    {
        TensorType tensor{Dev, {6}, {h2::DT::Any}};
        CHECK(tensor.is_contiguous());
        CHECK(is_chw_packed(tensor));
    }

    SECTION("Rank one, non-unit stride tensor")
    {
        TensorType tensor{
            Dev, mock_data, {6}, {h2::DT::Any}, {2}, h2::ComputeStream{Dev}};
        CHECK_FALSE(tensor.is_contiguous());
        CHECK_FALSE(is_chw_packed(tensor));
    }

    SECTION("Rank two, fully packed tensor")
    {
        TensorType tensor{Dev, {6, 4}, {h2::DT::Any, h2::DT::Any}};
        CHECK(tensor.is_contiguous());
        CHECK(is_chw_packed(tensor));
    }

    SECTION("Rank two, partially packed tensor")
    {
        TensorType tensor{Dev,
                          mock_data,
                          {6, 4},
                          {h2::DT::Any, h2::DT::Any},
                          {1, 8},
                          h2::ComputeStream{Dev}};
        CHECK_FALSE(tensor.is_contiguous());
        CHECK(is_chw_packed(tensor));
    }

    SECTION("Rank two, non-packed tensor")
    {
        TensorType tensor{Dev,
                          mock_data,
                          {6, 4},
                          {h2::DT::Any, h2::DT::Any},
                          {2, 12},
                          h2::ComputeStream{Dev}};
        CHECK_FALSE(tensor.is_contiguous());
        CHECK_FALSE(is_chw_packed(tensor));
    }

    SECTION("Rank 5, fully-packed tensor")
    {
        TensorType tensor{
            Dev,
            {7, 6, 5, 4, 3},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any}};
        CHECK(tensor.is_contiguous());
        CHECK(is_chw_packed(tensor));
    }

    SECTION("Rank 5, chw-packed tensor")
    {
        TensorType tensor{
            Dev,
            mock_data,
            {7, 6, 5, 4, 3},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any},
            {1, 7, 42, 210, 900},
            h2::ComputeStream{Dev}};
        CHECK_FALSE(tensor.is_contiguous());
        CHECK(is_chw_packed(tensor));
    }

    SECTION("Rank 5, non-packed tensor")
    {
        TensorType tensor{
            Dev,
            mock_data,
            {7, 6, 5, 4, 3},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any},
            {2, 14, 84, 420, 1680},
            h2::ComputeStream{Dev}};
        CHECK_FALSE(tensor.is_contiguous());
        CHECK_FALSE(is_chw_packed(tensor));

        tensor = TensorType{
            Dev,
            mock_data,
            {7, 6, 5, 4, 3},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any},
            {1, 8, 48, 240, 960},
            h2::ComputeStream{Dev}};
        CHECK_FALSE(tensor.is_contiguous());
        CHECK_FALSE(is_chw_packed(tensor));

        tensor = TensorType{
            Dev,
            mock_data,
            {7, 6, 5, 4, 3},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any},
            {1, 7, 43, 215, 860},
            h2::ComputeStream{Dev}};
        CHECK_FALSE(tensor.is_contiguous());
        CHECK_FALSE(is_chw_packed(tensor));

        tensor = TensorType{
            Dev,
            mock_data,
            {7, 6, 5, 4, 3},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any},
            {1, 7, 42, 211, 844},
            h2::ComputeStream{Dev}};
        CHECK_FALSE(tensor.is_contiguous());
        CHECK_FALSE(is_chw_packed(tensor));
    }
}

TEMPLATE_LIST_TEST_CASE("Hydrogen to DiHydrogen conversion",
                        "[tensor][h_h2]",
                        AllDevList)
{
    constexpr h2::Device Dev = TestType::value;
    constexpr hydrogen::Device HDev = h2::HydrogenDevice<Dev>;
    //using TensorType = h2::Tensor<DataType>;
    using MatrixType = El::Matrix<DataType, HDev>;

    // Shapes to use, when needed
    int const m = 6, n = 4, ldim = 9;

    SECTION("Empty matrix")
    {
        MatrixType mat;
        CHECK_THROWS(h2::as_h2_tensor(mat));
    }

    SECTION("Empty const matrix")
    {
        MatrixType const mat;
        CHECK_THROWS(h2::as_h2_tensor(mat));
    }

    SECTION("Column vector")
    {
        auto mat = make_matrix<HDev>(m, 1);
        auto tensor = h2::as_h2_tensor(mat);

        CHECK(tensor.ndim() == 1);
        CHECK(tensor.is_contiguous());
        CHECK(tensor.is_view());
        CHECK(tensor.get_view_type() == h2::ViewType::Mutable);

        CHECK(tensor.shape() == h2::ShapeTuple{m});
        CHECK(tensor.strides() == h2::StrideTuple{1});

        for (int row = 0; row < m; ++row)
        {
            INFO("ROW=" << row);
            CHECK(tensor.get({row}) == mat.LockedBuffer(row, 0));
        }
    }

    SECTION("Const column vector")
    {
        auto const mat = make_matrix<HDev>(m, 1);
        auto tensor = h2::as_h2_tensor(mat);

        CHECK(tensor.ndim() == 1);
        CHECK(tensor.is_contiguous());
        CHECK(tensor.is_view());
        CHECK(tensor.is_const_view());
        CHECK(tensor.get_view_type() == h2::ViewType::Const);

        CHECK(tensor.shape() == h2::ShapeTuple{m});
        CHECK(tensor.strides() == h2::StrideTuple{1});

        for (int row = 0; row < m; ++row)
        {
            INFO("ROW=" << row);
            CHECK(tensor.const_get({row}) == mat.LockedBuffer(row, 0));
        }
    }

    SECTION("Row vector")
    {
        auto mat = make_matrix<HDev>(1, n, ldim);
        auto tensor = h2::as_h2_tensor(mat);
        CHECK(tensor.ndim() == 1);
        CHECK_FALSE(tensor.is_contiguous());
        CHECK(tensor.is_view());
        CHECK(tensor.get_view_type() == h2::ViewType::Mutable);

        CHECK(tensor.shape() == h2::ShapeTuple{n});
        CHECK(tensor.strides() == h2::StrideTuple{ldim});

        for (int col = 0; col < n; ++col)
        {
            INFO("COL=" << col);
            CHECK(tensor.get({col}) == mat.LockedBuffer(0, col));
        }
    }

    SECTION("Const row vector")
    {
        auto const mat = make_matrix<HDev>(1, n);
        auto tensor = h2::as_h2_tensor(mat);
        CHECK(tensor.ndim() == 1);
        CHECK(tensor.is_contiguous());
        CHECK(tensor.is_view());
        CHECK(tensor.is_const_view());
        CHECK(tensor.get_view_type() == h2::ViewType::Const);

        CHECK(tensor.shape() == h2::ShapeTuple{n});
        CHECK(tensor.strides() == h2::StrideTuple{1});

        for (int col = 0; col < n; ++col)
        {
            INFO("COL=" << col);
            CHECK(tensor.const_get({col}) == mat.LockedBuffer(0, col));
        }
    }

    SECTION("Contiguous matrix")
    {
        auto mat = make_matrix<HDev>(m, n);
        auto tensor = h2::as_h2_tensor(mat);
        CHECK(tensor.ndim() == 2);
        CHECK(tensor.is_contiguous());
        CHECK(tensor.is_view());
        CHECK(tensor.get_view_type() == h2::ViewType::Mutable);

        CHECK(tensor.shape() == h2::ShapeTuple{m, n});
        CHECK(tensor.strides() == h2::StrideTuple{1, m});

        for (int col = 0; col < n; ++col)
            for (int row = 0; row < m; ++row)
            {
                INFO("ROW=" << row << ", COL=" << col);
                CHECK(tensor.get({row, col}) == mat.LockedBuffer(row, col));
            }
    }

    SECTION("Contiguous const matrix")
    {
        auto const mat = make_matrix<HDev>(m, n);
        auto tensor = h2::as_h2_tensor(mat);
        CHECK(tensor.ndim() == 2);
        CHECK(tensor.is_contiguous());
        CHECK(tensor.is_view());
        CHECK(tensor.is_const_view());
        CHECK(tensor.get_view_type() == h2::ViewType::Const);

        CHECK(tensor.shape() == h2::ShapeTuple{m, n});
        CHECK(tensor.strides() == h2::StrideTuple{1, m});

        for (int col = 0; col < n; ++col)
            for (int row = 0; row < m; ++row)
            {
                INFO("ROW=" << row << ", COL=" << col);
                CHECK(tensor.const_get({row, col}) == mat.LockedBuffer(row, col));
            }
    }

    SECTION("Non-contiguous matrix")
    {
        auto mat = make_matrix<HDev>(m, n, ldim);
        auto tensor = h2::as_h2_tensor(mat);
        CHECK(tensor.ndim() == 2);
        CHECK_FALSE(tensor.is_contiguous());
        CHECK(tensor.is_view());
        CHECK(tensor.get_view_type() == h2::ViewType::Mutable);

        CHECK(tensor.shape() == h2::ShapeTuple{m, n});
        CHECK(tensor.strides() == h2::StrideTuple{1, ldim});

        for (int col = 0; col < n; ++col)
            for (int row = 0; row < m; ++row)
            {
                INFO("ROW=" << row << ", COL=" << col);
                CHECK(tensor.get({row, col}) == mat.LockedBuffer(row, col));
            }
    }

    SECTION("Non-contiguous const matrix")
    {
        auto const mat = make_matrix<HDev>(m, n, ldim);
        auto tensor = h2::as_h2_tensor(mat);
        CHECK(tensor.ndim() == 2);
        CHECK_FALSE(tensor.is_contiguous());
        CHECK(tensor.is_view());
        CHECK(tensor.is_const_view());
        CHECK(tensor.get_view_type() == h2::ViewType::Const);

        CHECK(tensor.shape() == h2::ShapeTuple{m, n});
        CHECK(tensor.strides() == h2::StrideTuple{1, ldim});

        for (int col = 0; col < n; ++col)
            for (int row = 0; row < m; ++row)
            {
                INFO("ROW=" << row << ", COL=" << col);
                CHECK(tensor.const_get({row, col}) == mat.LockedBuffer(row, col));
            }
    }
}

TEMPLATE_LIST_TEST_CASE("DiHydrogen to Hydrogen conversion",
                        "[tensor][h2_h]",
                        AllDevList)
{
    constexpr h2::Device Dev = TestType::value;
    //constexpr hydrogen::Device HDev = h2::HydrogenDevice<Dev>;
    using TensorType = h2::Tensor<DataType>;
    //using MatrixType = El::Matrix<DataType, HDev>;

    // Metadata checks ONLY
    DataType* const mock_data = reinterpret_cast<DataType* const>(1);
    DataType const* const mock_const_data = reinterpret_cast<DataType const* const>(1);

    SECTION("Rank-1 contiguous tensor")
    {
        auto tensor = TensorType{Dev, {9}, {h2::DT::Any}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{9});
        CHECK(mat.Width() == El::Int{1});
        CHECK(mat.LDim() == El::Int{9});
        CHECK(mat.Viewing());
        CHECK_FALSE(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-1 const contiguous tensor")
    {
        auto const tensor = TensorType{Dev, {9}, {h2::DT::Any}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{9});
        CHECK(mat.Width() == El::Int{1});
        CHECK(mat.LDim() == El::Int{9});
        CHECK(mat.Viewing());
        CHECK(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-1 non-contiguous tensor")
    {
        auto tensor = TensorType{
            Dev, mock_data, {9}, {h2::DT::Any}, {2}, h2::ComputeStream{Dev}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{1});
        CHECK(mat.Width() == El::Int{9});
        CHECK(mat.LDim() == El::Int{2});
        CHECK(mat.Viewing());
        CHECK_FALSE(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-1 const non-contiguous tensor")
    {
        auto const tensor = TensorType{Dev,
                                       mock_const_data,
                                       {9},
                                       {h2::DT::Any},
                                       {7},
                                       h2::ComputeStream{Dev}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{1});
        CHECK(mat.Width() == El::Int{9});
        CHECK(mat.LDim() == El::Int{7});
        CHECK(mat.Viewing());
        CHECK(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-2 contiguous tensor")
    {
        auto tensor = TensorType{Dev, {3, 9}, {h2::DT::Any, h2::DT::Any}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{3});
        CHECK(mat.Width() == El::Int{9});
        CHECK(mat.LDim() == El::Int{3});
        CHECK(mat.Viewing());
        CHECK_FALSE(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-2 const contiguous tensor")
    {
        auto const tensor = TensorType{Dev, {4, 8}, {h2::DT::Any, h2::DT::Any}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{4});
        CHECK(mat.Width() == El::Int{8});
        CHECK(mat.LDim() == El::Int{4});
        CHECK(mat.Viewing());
        CHECK(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-2 CHW-packed tensor")
    {
        auto tensor = TensorType{Dev,
                                 mock_data,
                                 {4, 8},
                                 {h2::DT::Any, h2::DT::Any},
                                 {1, 6},
                                 h2::ComputeStream{Dev}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{4});
        CHECK(mat.Width() == El::Int{8});
        CHECK(mat.LDim() == El::Int{6});
        CHECK(mat.Viewing());
        CHECK_FALSE(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-2 const CHW-packed tensor")
    {
        auto const tensor = TensorType{Dev,
                                       mock_const_data,
                                       {4, 8},
                                       {h2::DT::Any, h2::DT::Any},
                                       {1, 7},
                                       h2::ComputeStream{Dev}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{4});
        CHECK(mat.Width() == El::Int{8});
        CHECK(mat.LDim() == El::Int{7});
        CHECK(mat.Viewing());
        CHECK(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-2 non-CHW-packed tensor")
    {
        auto tensor = TensorType{Dev,
                                 mock_data,
                                 {4, 8},
                                 {h2::DT::Any, h2::DT::Any},
                                 {2, 10},
                                 h2::ComputeStream{Dev}};
        CHECK_THROWS(h2::as_h_mat<Dev>(tensor));
    }

    SECTION("Rank-2 non-CHW-packed const tensor")
    {
        auto const tensor = TensorType{Dev,
                                       mock_const_data,
                                       {4, 8},
                                       {h2::DT::Any, h2::DT::Any},
                                       {3, 15},
                                       h2::ComputeStream{Dev}};
        CHECK_THROWS(h2::as_h_mat<Dev>(tensor));
    }

    SECTION("Rank-3 fully-packed tensor")
    {
        auto tensor =
          TensorType{Dev, {3, 4, 5}, {h2::DT::Any, h2::DT::Any, h2::DT::Any}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{12});
        CHECK(mat.Width() == El::Int{5});
        CHECK(mat.LDim() == El::Int{12});
        CHECK(mat.Viewing());
        CHECK_FALSE(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-3 fully-packed const tensor")
    {
        auto const tensor =
          TensorType{Dev, {3, 4, 5}, {h2::DT::Any, h2::DT::Any, h2::DT::Any}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{12});
        CHECK(mat.Width() == El::Int{5});
        CHECK(mat.LDim() == El::Int{12});
        CHECK(mat.Viewing());
        CHECK(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-3 CHW-packed tensor")
    {
        auto tensor = TensorType{Dev,
                                 mock_data,
                                 {3, 4, 5},
                                 {h2::DT::Any, h2::DT::Any, h2::DT::Any},
                                 {1, 3, 20},
                                 h2::ComputeStream{Dev}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{12});
        CHECK(mat.Width() == El::Int{5});
        CHECK(mat.LDim() == El::Int{20});
        CHECK(mat.Viewing());
        CHECK_FALSE(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-3 CHW-packed const tensor")
    {
        auto const tensor = TensorType{Dev,
                                       mock_const_data,
                                       {3, 4, 5},
                                       {h2::DT::Any, h2::DT::Any, h2::DT::Any},
                                       {1, 3, 13},
                                       h2::ComputeStream{Dev}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{12});
        CHECK(mat.Width() == El::Int{5});
        CHECK(mat.LDim() == El::Int{13});
        CHECK(mat.Viewing());
        CHECK(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-3 non-packed tensor")
    {
        auto tensor = TensorType{Dev,
                                 mock_data,
                                 {3, 4, 5},
                                 {h2::DT::Any, h2::DT::Any, h2::DT::Any},
                                 {2, 7, 40},
                                 h2::ComputeStream{Dev}};
        CHECK_THROWS(h2::as_h_mat<Dev>(tensor));
    }

    SECTION("Rank-3 non-packed const tensor")
    {
        auto const tensor = TensorType{Dev, mock_const_data,
                                       {3, 4, 5},
                                       {h2::DT::Any, h2::DT::Any, h2::DT::Any},
                                       {1, 8, 33}, h2::ComputeStream{Dev}};
        CHECK_THROWS(h2::as_h_mat<Dev>(tensor));
    }

    SECTION("Rank-5 fully-packed tensor")
    {
        auto tensor = TensorType{
            Dev,
            {2, 3, 4, 5, 6},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{120});
        CHECK(mat.Width() == El::Int{6});
        CHECK(mat.LDim() == El::Int{120});
        CHECK(mat.Viewing());
        CHECK_FALSE(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-5 fully-packed const tensor")
    {
        auto const tensor = TensorType{
          Dev,
          {2, 3, 4, 5, 6},
          {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{120});
        CHECK(mat.Width() == El::Int{6});
        CHECK(mat.LDim() == El::Int{120});
        CHECK(mat.Viewing());
        CHECK(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-5 CHW-packed tensor")
    {
        auto tensor = TensorType{
            Dev,
            mock_data,
            {2, 3, 4, 5, 6},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any},
            {1, 2, 6, 24, 123},
            h2::ComputeStream{Dev}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{120});
        CHECK(mat.Width() == El::Int{6});
        CHECK(mat.LDim() == El::Int{123});
        CHECK(mat.Viewing());
        CHECK_FALSE(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-5 CHW-packed const tensor")
    {
        auto const tensor = TensorType{
            Dev,
            mock_const_data,
            {2, 3, 4, 5, 6},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any},
            {1, 2, 6, 24, 121},
            h2::ComputeStream{Dev}};
        auto mat = h2::as_h_mat<Dev>(tensor);

        CHECK(mat.Height() == El::Int{120});
        CHECK(mat.Width() == El::Int{6});
        CHECK(mat.LDim() == El::Int{121});
        CHECK(mat.Viewing());
        CHECK(mat.Locked());
        CHECK(mat.LockedBuffer() == tensor.const_data());
    }

    SECTION("Rank-5 non-packed tensor")
    {
        auto tensor = TensorType{
            Dev,
            mock_data,
            {2, 3, 4, 5, 6},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any},
            {3, 6, 18, 72, 360},
            h2::ComputeStream{Dev}};
        CHECK_THROWS(h2::as_h_mat<Dev>(tensor));
    }

    SECTION("Rank-5 non-packed const tensor")
    {
        auto const tensor = TensorType{
            Dev,
            mock_const_data,
            {2, 3, 4, 5, 6},
            {h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any, h2::DT::Any},
            {1, 2, 6, 25, 240},
            h2::ComputeStream{Dev}};
        CHECK_THROWS(h2::as_h_mat<Dev>(tensor));
    }
}
