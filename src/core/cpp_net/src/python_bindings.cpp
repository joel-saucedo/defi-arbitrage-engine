/**
 * python bindings for mev network monitor
 * provides high-level interface to c++ optimized networking
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <memory>
#include <vector>
#include <string>

// include the network monitor implementation
#include "network_monitor.cpp"

namespace py = pybind11;

PYBIND11_MODULE(mev_net_py, m) {
    m.doc() = "high-performance mev network monitoring";
    
    // transaction structure
    py::class_<mev::Transaction>(m, "Transaction")
        .def_readonly("timestamp_ns", &mev::Transaction::timestamp_ns)
        .def_readonly("block_number", &mev::Transaction::block_number)
        .def_readonly("gas_price", &mev::Transaction::gas_price)
        .def_readonly("gas_limit", &mev::Transaction::gas_limit)
        .def_property_readonly("hash", [](const mev::Transaction& tx) {
            return std::string(tx.hash);
        })
        .def_property_readonly("from_address", [](const mev::Transaction& tx) {
            return std::string(tx.from);
        })
        .def_property_readonly("to_address", [](const mev::Transaction& tx) {
            return std::string(tx.to);
        });
    
    // websocket client
    py::class_<mev::WebSocketClient>(m, "WebSocketClient")
        .def(py::init<>())
        .def("connect", &mev::WebSocketClient::connect)
        .def("disconnect", &mev::WebSocketClient::disconnect)
        .def("send_message", &mev::WebSocketClient::send_message)
        .def("get_messages_received", &mev::WebSocketClient::get_messages_received)
        .def("get_bytes_received", &mev::WebSocketClient::get_bytes_received)
        .def("get_avg_latency_ns", &mev::WebSocketClient::get_avg_latency_ns)
        .def("get_buffer_size", &mev::WebSocketClient::get_buffer_size);
    
    // mempool monitor
    py::class_<mev::MempoolMonitor>(m, "MempoolMonitor")
        .def(py::init<uint64_t, uint64_t>(), 
             py::arg("min_gas") = 1000000000, 
             py::arg("max_gas") = 1000000000000)
        .def("add_connection", &mev::MempoolMonitor::add_connection)
        .def("start", &mev::MempoolMonitor::start)
        .def("stop", &mev::MempoolMonitor::stop)
        .def("add_target_address", &mev::MempoolMonitor::add_target_address);
}
