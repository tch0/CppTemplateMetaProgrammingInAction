#pragma once

#include <policy/policy_container.hpp>
#include <data/matrix/matrix.hpp>
#include <facility/var_type_dict.hpp>
#include <facility/data_copy.hpp>
#include <param_initializer/init_policy.hpp>
#include <type_traits>
#include <string>
#include <map>
#include <stdexcept>

namespace MetaNN
{

namespace NsParamInitializer
{

// 将传入的初始化策略中的模板参数作为VarTypeDict的模板实参，并且去重之后得到输出类型
// 仅针对: PInitializerIs, PWeightInitializerIs, PBiasInitializerIs
// MetaNN中有去重逻辑，这里没有加，暂时没看到去重的必要性。
template<typename TRes, typename... TPolicies>
struct FillerTagsFromPolicy_
{
    using type = TRes;
};

template<typename... TRes, typename TCur, typename... TRest>
struct FillerTagsFromPolicy_<VarTypeDict<TRes...>, PInitializerIs<TCur>, TRest...>
    : FillerTagsFromPolicy_<VarTypeDict<TRes..., TCur>, TRest...> {};

template<typename... TRes, typename TCur, typename... TRest>
struct FillerTagsFromPolicy_<VarTypeDict<TRes...>, PWeightInitializerIs<TCur>, TRest...>
    : FillerTagsFromPolicy_<VarTypeDict<TRes..., TCur>, TRest...> {};

template<typename... TRes, typename TCur, typename... TRest>
struct FillerTagsFromPolicy_<VarTypeDict<TRes...>, PBiasInitializerIs<TCur>, TRest...>
    : FillerTagsFromPolicy_<VarTypeDict<TRes..., TCur>, TRest...> {};

template<typename... TRes, typename... TSub, typename... TRest>
struct FillerTagsFromPolicy_<VarTypeDict<TRes...>, SubPolicyContainer<TSub...>, TRest...>
    : FillerTagsFromPolicy_<VarTypeDict<TRes...>, TSub..., TRest...> {};

} // namespace NsParamInitializer

template<typename... TPolicies>
using FillerTags2NamedParams = typename NsParamInitializer::FillerTagsFromPolicy_<VarTypeDict<>, TPolicies...>::type;

// TFiller是一个VarTypeDict类型
template<typename TElem, typename TPolicyContainer, typename TFillers>
class ParamInitializer
{
public:
    using PolicyCont = TPolicyContainer;

    ParamInitializer(TFillers&& filler)
        : m_filler(std::move(filler))
    {
    }

    // 初始化器的设置与获取
    template<typename TTag, typename TVal>
    auto setFiller(TVal&& val) &&
    {
        auto newFiller = std::move(m_filler).template set<TTag, TVal>(std::forward<TVal>(val));
        using newFillerType = std::remove_cvref_t<decltype(newFiller)>;
        return ParamInitializer<TElem, TPolicyContainer, newFillerType>(std::move(newFiller));
    }
    template<typename TTag, typename val>
    auto getFiller()
    {
        return m_filler.template get<TTag>();
    }

    // 参数矩阵的设置与获取
    template<typename TElem2, typename TDevice2>
    void setMatrix(const std::string& name, const Matrix<TElem2, TDevice2>& param)
    {
        if (m_params.find(name) != m_params.end())
        {
            throw std::runtime_error("Duplicate parameter matrix: " + name);
        }
        m_params.insert({name, param});
    }
    // 通过深拷贝方式获取参数矩阵，不会共享内存，从ParamInitializer获取的参数矩阵不会共享数据
    template<typename TElem2, typename TDevice2>
    void getMatrix(const std::string& name, Matrix<TElem2, TDevice2>& res) const
    {
        auto it = m_params.find(name);
        if (it == m_params.end())
        {
            throw std::runtime_error("Parameter no exist: " + name);
        }
        const auto& mat = it->second;
        if (mat.rowNum() != res.rowNum() || mat.colNum() != res.colNum())
        {
            throw std::runtime_error("Matrices dimension mismatch!");
        }
        dataCopy(mat, res);
    }
    bool IsMatrixExist(const std::string& name) const
    {
        return m_params.find(name) != m_params.end();
    }

private:
    TFillers m_filler;
    std::map<std::string, Matrix<TElem, DeviceTags::CPU>> m_params;
};

template<typename TElem, typename... TPolicies>
auto makeInitializer()
{
    using DictType = FillerTags2NamedParams<TPolicies...>; // 获取传入的策略对象标签并构造出对应参数的VarTypeDict类型
    using FillerDictType = std::remove_cvref_t<decltype(DictType::create())>; // 构造异类词典对象类型VarTypeDict::Values<NullParameter...>
    return PramInitializer<TElem, PolicyContainer<TPolicies...>, FillerDictType>(DictType::create()); // 构造出参数初始化器对象
}

} // namespace MetaNN