#define TypePolicyObj(PolicyName, Major, Minor, Value)\
struct PolicyName : virtual public Major\
{\
    using MinorClass = Major::Minor##TypeCategory;\
    using Minor = Major::Minor##TypeCategory::Value;\
}

#define ValuePolicyObj(PolicyName, Major, Minor, Value)\
struct PolicyName : virtual public Major\
{\
    using MinorClass = Major::Minor##ValueCategory;\
private:\
    using type = std::remove_cvref_t<decltype(Major::Minor)>;\
public:\
    static constexpr type Minor = static_cast<type>(Value);\
}

#define TypePolicyTemplate(PolicyName, Major, Minor)\
template<typename T>\
struct PolicyName : virtual public Major\
{\
    using MinorClass = Major::Minor##TypeCategory;\
    using Minor = T;\
}

#define ValuePolicyTemplate(PolicyName, Major, Minor)\
template<std::remove_cvref_t<decltype(Major::Minor)> T>\
struct PolicyName : virtual public Major\
{\
    using MinorClass = Major::Minor##ValueCategory;\
private:\
    using type = std::remove_cvref_t<decltype(Major::Minor)>;\
public:\
    static constexpr type Minor = T;\
}
