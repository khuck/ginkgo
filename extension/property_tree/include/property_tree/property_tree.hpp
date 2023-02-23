#ifndef GKO_EXT_PROPERT_TREE_HPP_
#define GKO_EXT_PROPERT_TREE_HPP_

#include <deque>
#include <exception>
#include <list>
#include <string>
#include <type_traits>

#include "property_tree/data.hpp"

namespace gko {
namespace extension {


template <typename data_type>
class pnode_s {
public:
    using key_type = std::string;

    enum status_t { empty, array, object, object_list };

    pnode_s(const std::string& name = "root")
        : name_(name), status_(status_t::empty)
    {}

    pnode_s(const std::string& name, const data_type& data)
        : name_(name), status_(status_t::object), data_(data)
    {}

    // Get the content of node
    data_type get() const
    {
        assert(status_ == status_t::object);
        return data_;
    }

    // Get the content of node via path (. as separator)
    data_type get(const std::string& path) const
    {
        return this->get_child(path).get();
    }

    // Get the data of node's content
    template <typename T>
    T get() const
    {
        assert(status_ == status_t::object);
        return data_.template get<T>();
    }

    // Get the data of node's content via path (. as separator)
    template <typename T>
    T get(const std::string& path) const
    {
        return this->get_child(path).template get<T>();
    }

    // Get the list of children. It's only available for const
    const std::deque<pnode_s<data_type>>& get_child_list() const
    {
        assert(children_.size() > 0);
        return children_;
    }

    // Get the child by given path (. as separator)
    pnode_s<data_type>& get_child(const std::string& path)
    {
        auto sep = path.find(".");
        if (sep == std::string::npos) {
            return children_.at(key_map_.at(path));
        }
        return children_.at(key_map_.at(path.substr(0, sep)))
            .get_child(path.substr(sep + 1));
    }

    const pnode_s<data_type>& get_child(const std::string& path) const
    {
        auto sep = path.find(".");
        if (sep == std::string::npos) {
            return children_.at(key_map_.at(path));
        }
        return children_.at(key_map_.at(path.substr(0, sep)))
            .get_child(path.substr(sep + 1));
    }

    // Get the index i children
    pnode_s<data_type>& get_child(int i) { return children_.at(i); }

    // Check the status
    bool is(status_t s) const { return this->get_status() == s; }

    int get_size() const { return children_.size(); }

    status_t get_status() const { return status_; }

    std::string get_name() const { return name_; }

    // Only allow change the status from empty
    void update_status(status_t status)
    {
        if (status_ != status_t::empty && status_ != status) {
            throw std::runtime_error("Can not change the status");
        }
        if (status == status_t::empty) {
            throw std::runtime_error("Can not clear");
        }
        status_ = status;
    }

    void set(const data_type& data)
    {
        update_status(status_t::object);
        data_ = data;
    }

    void insert(const std::string& name, const data_type& data)
    {
        this->update_status(status_t::object_list);
        this->insert_item(name, data);
    }

    // insert for array
    void insert(const data_type& data)
    {
        this->update_status(status_t::array);
        int size = key_map_.size();
        std::string name = "array_" + std::to_string(size);
        this->insert_item(name, data);
    }

    void insert(std::initializer_list<data_type> data)
    {
        this->update_status(status_t::array);
        for (auto item : data) {
            this->insert(item);
        }
    }

    void allocate(const std::string& name)
    {
        this->update_status(status_t::object_list);
        this->insert_item(name);
    }

    void allocate_array(int num)
    {
        this->update_status(status_t::array);
        auto size = children_.size();
        for (int i = 0; i < num; i++) {
            std::string name = "array_" + std::to_string(size + i);
            this->insert_item(name);
        }
    }

private:
    // It's for insert item, please check the status before it
    void insert_item(const std::string& name, const data_type& data)
    {
        int size = key_map_.size();
        auto p = key_map_.emplace(name, size);
        if (!p.second) {
            throw std::runtime_error("Have the same key");
        }
        children_.emplace_back(pnode_s{name, data});
    }

    // insert empty
    void insert_item(const std::string& name)
    {
        int size = key_map_.size();
        auto p = key_map_.emplace(name, size);
        if (!p.second) {
            throw std::runtime_error("Have the same key");
        }
        children_.emplace_back(pnode_s{name});
    }

    std::string name_;
    std::deque<pnode_s<data_type>> children_;
    std::map<key_type, int> key_map_;
    data_type data_;
    status_t status_;
};

using pnode = pnode_s<data_s>;


}  // namespace extension
}  // namespace gko


#endif