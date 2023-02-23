#ifndef GKO_EXT_PROPERT_TREE_DATA_HPP_
#define GKO_EXT_PROPERT_TREE_DATA_HPP_

#include <deque>
#include <exception>
#include <list>
#include <string>
#include <type_traits>

namespace gko {
namespace extension {
class data_s {
public:
    enum tag_type { str_t, int_t, double_t, bool_t, empty_t };

    template <tag_type>
    struct tag {
        using type = void;
    };

    /**
     * Get the data with Type T
     */
    template <typename T>
    T get() const
    {
        throw std::runtime_error("Not Supported");
    }

    template <tag_type T>
    typename tag<T>::type get() const;

    tag_type get_tag() const { return tag_; }

    /**
     * Default Constructor
     */
    data_s() : tag_(tag_type::empty_t) {}

    /**
     * Constructor for bool
     */
    data_s(bool bb) : tag_(tag_type::bool_t) { bool_ = bb; }

    /**
     * Constructor for integer
     */
    data_s(long long int ii) : tag_(tag_type::int_t) { int_ = ii; }

    /**
     * Constructor for integer with all integer type
     */
    template <typename T, typename = typename std::enable_if<
                              std::is_integral<T>::value>::type>
    data_s(T ii) : data_s(static_cast<long long int>(ii))
    {}

    /**
     * Constructor for string
     */
    data_s(const std::string& str) : tag_(tag_type::str_t) { str_ = str; };

    /**
     * Constructor for char (otherwise, it will use bool)
     */
    data_s(const char* s) : data_s(std::string(s)) {}

    /**
     * Constructor for double
     */
    data_s(double dd) : tag_(tag_type::double_t) { double_ = dd; };

    /**
     * Constructor for float
     */
    data_s(float dd) : data_s(static_cast<double>(dd)) {}

    /**
     * Overloading stream operator<<
     */
    friend std::ostream& operator<<(std::ostream& stream, const data_s& data)
    {
        switch (data.tag_) {
        case tag_type::empty_t:
            stream << "empty";
            break;
        case tag_type::bool_t:
            stream << (data.bool_ ? "true" : "false");
            break;
        case tag_type::int_t:
            stream << data.int_;
            break;
        case tag_type::str_t:
            stream << data.str_;
            break;
        case tag_type::double_t:
            stream << data.double_;
            break;
        }
        return stream;
    }

private:
    tag_type tag_;
    std::string str_;
    long long int int_;
    double double_;
    bool bool_;
};

template <>
long long int data_s::get<long long int>() const
{
    assert(tag_ == data_s::tag_type::int_t);
    return int_;
}

template <>
std::string data_s::get<std::string>() const
{
    assert(tag_ == data_s::tag_type::str_t);
    return str_;
}

template <>
double data_s::get<double>() const
{
    assert(tag_ == data_s::tag_type::double_t);
    return double_;
}

template <>
bool data_s::get<bool>() const
{
    assert(tag_ == data_s::tag_type::bool_t);
    return bool_;
}


template <>
struct data_s::tag<data_s::tag_type::bool_t> {
    using type = bool;
};

template <>
struct data_s::tag<data_s::tag_type::int_t> {
    using type = long long int;
};

template <>
struct data_s::tag<data_s::tag_type::double_t> {
    using type = double;
};

template <>
struct data_s::tag<data_s::tag_type::str_t> {
    using type = std::string;
};


}  // namespace extension
}  // namespace gko


#endif  // GKO_EXT_PROPERT_TREE_DATA_HPP_