#pragma once

#include <data/tags.hpp>
#include <type_traits>
#include <concepts>

namespace MetaNN
{

// 前向声明
template<typename TElem, typename TDevice> class Scalar;
template<typename TElem, typename TDevice> class Matrix;
template<typename TElem, typename TDevice, typename TCategory> class Batch;

// 主体类型
template<typename TCategory, typename TElem, typename TDevice>
struct PrincipalDataType_;

template<typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::Scalar, TElem, TDevice>
{
    using type = Scalar<TElem, TDevice>;
};

template<typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::Matrix, TElem, TDevice>
{
    using type = Matrix<TElem, TDevice>;
};

template<typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchScalar, TElem, TDevice>
{
    using type = Batch<TElem, TDevice, CategoryTags::Scalar>;
};

template<typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchMatrix, TElem, TDevice>
{
    using type = Batch<TElem, TDevice, CategoryTags::Matrix>;
};

template<typename TCategory, typename TElem, typename TDevice>
using PrincipalDataType = typename PrincipalDataType_<TCategory, TElem, TDevice>::type;


// 获取一个类型的类别，除了可以定义category嵌套类型以声明类别，也可以通过特化DataCategory_非侵入式声明
template<typename T> requires requires { typename T::Category; }
struct DataCategory_
{
    using type = typename T::Category;
};
// 对const和引用偏特化，使其更加通用
template<typename T>
struct DataCategory_<const T> : DataCategory_<T> {};
template<typename T>
struct DataCategory_<T&> : DataCategory_<T> {};
template<typename T>
struct DataCategory_<T&&> : DataCategory_<T> {};

template<typename T>
using DataCategory = typename DataCategory_<T>::type;

// 合法数据类型的约束
template<typename TDataType>
concept ValidDataTypeC = requires
{
    typename TDataType::ElementType;
    typename TDataType::DeviceType;
};

template<typename TDataType>
concept ValidMatrixTypeC = requires(const TDataType& data)
{
    { data.rowNum() } -> std::same_as<std::size_t>;
    { data.colNum() } -> std::same_as<std::size_t>;
};

template<typename TDataType>
concept ValidBatchTypeC = requires(const TDataType& data)
{
    { data.batchNum() } -> std::same_as<std::size_t>;
};

template<typename TDataType>
concept ValidEvaluationTypeC = requires(TDataType data)
{
    data;
    // todo yet!
};

// 类别判断概念
// Scalar
template<typename T>
concept IsScalarC = std::is_same_v<DataCategory<T>, CategoryTags::Scalar> && ValidDataTypeC<T>;

// Matrix
template<typename T>
concept IsMatrixC = std::is_same_v<DataCategory<T>, CategoryTags::Matrix> && ValidDataTypeC<T> && ValidMatrixTypeC<T>;

// BatchScalar
template<typename T>
concept IsBatchScalarC = std::is_same_v<DataCategory<T>, CategoryTags::BatchScalar> && ValidDataTypeC<T> && ValidBatchTypeC<T>;

// BatchMatrix
template<typename T>
concept IsBatchMatrixC = std::is_same_v<DataCategory<T>, CategoryTags::BatchMatrix> && ValidDataTypeC<T> && ValidBatchTypeC<T> && ValidMatrixTypeC<T>;

// 范围更广的类别判断概念，对引用和const修饰的复合类型则根据其底层类型判断
template<typename T>
concept ScalarC = IsScalarC<std::remove_cvref_t<T>>;

template<typename T>
concept MatrixC = IsMatrixC<std::remove_cvref_t<T>>;

template<typename T>
concept BatchScalarC = IsBatchScalarC<std::remove_cvref_t<T>>;

template<typename T>
concept BatchMatrixC = IsBatchMatrixC<std::remove_cvref_t<T>>;

// 用于static_assert
template<typename T>
constexpr bool DependencyFalse = false;

} // namespace MetaNN