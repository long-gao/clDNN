#include <../api/CPP/cldnn_defs.h>
#include <../api/CPP/engine.hpp>
#include <../api/CPP/input_layout.hpp>
#include <../api/CPP/memory.hpp>
#include <../api/CPP/data.hpp>
#include <../api/CPP/topology.hpp>
#include <../api/CPP/network.hpp>
#include <../api/CPP/convolution.hpp>
#include <iostream>
#include <string>
#include <chrono>

#include "helper_functions.h"

double run_conv(tensor& input_tensor, tensor& weights_tensor, tensor& biases_tensor, tensor& stride_tensor, tensor& padding_tensor, std::string data_type);
tensor create_tensor(int batch, int feature, int x, int y);

int main()
{
    int iter = 10;
    double exe_time = 10;
    std::string data_type = "float32"; // "float32" or "float16"
    int batch = 1, in_size = 224, channel = 3, filter = 64, kernel = 7, padding = 3, stride = 2;

    std::cout << "conv layer (1, 224, 3, 64, 7, 3, 2)"<< std::endl;
    batch = 1, in_size = 224, channel = 3, filter = 64, kernel = 7, padding = 3, stride = 2;
    tensor input_tensor = create_tensor(batch, channel, in_size, in_size);
    tensor weights_tensor = create_tensor(filter, channel, kernel, kernel);
    tensor biases_tensor = create_tensor(1, 1, filter, 1);
    tensor stride_tensor = create_tensor(1, 1, stride, stride);
    tensor padding_tensor = create_tensor(0, 0, -padding, -padding);
    exe_time = 0;
    for (int i = 0; i < iter; i++)
    {
        exe_time += run_conv(input_tensor, weights_tensor, biases_tensor, stride_tensor, padding_tensor, data_type);
    }
    std::cout << "Executing time: " << (exe_time / iter) << " milliseconds"<< std::endl;

    std::cout << "conv layer (1, 56, 64, 64, 3, 1, 1)"<< std::endl;
    batch = 1, in_size = 56, channel = 64, filter = 64, kernel = 3, padding = 1, stride = 1;
    input_tensor = create_tensor(batch, channel, in_size, in_size);
    weights_tensor = create_tensor(filter, channel, kernel, kernel);
    biases_tensor = create_tensor(1, 1, filter, 1);
    stride_tensor = create_tensor(1, 1, stride, stride);
    padding_tensor = create_tensor(0, 0, -padding, -padding);
    exe_time = 0;
    for (int i = 0; i < iter; i++)
    {
        exe_time += run_conv(input_tensor, weights_tensor, biases_tensor, stride_tensor, padding_tensor, data_type);
    }
    std::cout << "Executing time: " << (exe_time / iter) << " milliseconds"<< std::endl;

    std::cout << "conv layer (1, 28, 128, 128, 3, 1, 1)"<< std::endl;
    batch = 1, in_size = 28, channel = 128, filter = 128, kernel = 3, padding = 1, stride = 1;
    input_tensor = create_tensor(batch, channel, in_size, in_size);
    weights_tensor = create_tensor(filter, channel, kernel, kernel);
    biases_tensor = create_tensor(1, 1, filter, 1);
    stride_tensor = create_tensor(1, 1, stride, stride);
    padding_tensor = create_tensor(0, 0, -padding, -padding);
    exe_time = 0;
    for (int i = 0; i < iter; i++)
    {
        exe_time += run_conv(input_tensor, weights_tensor, biases_tensor, stride_tensor, padding_tensor, data_type);
    }
    std::cout << "Executing time: " << (exe_time / iter) << " milliseconds"<< std::endl;
    
    std::cout << "conv layer (1, 14, 256, 256, 3, 1, 1)"<< std::endl;
    batch = 1, in_size = 14, channel = 256, filter = 256, kernel = 3, padding = 1, stride = 1;
    input_tensor = create_tensor(batch, channel, in_size, in_size);
    weights_tensor = create_tensor(filter, channel, kernel, kernel);
    biases_tensor = create_tensor(1, 1, filter, 1);
    stride_tensor = create_tensor(1, 1, stride, stride);
    padding_tensor = create_tensor(0, 0, -padding, -padding);
    exe_time = 0;
    for (int i = 0; i < iter; i++)
    {
        exe_time += run_conv(input_tensor, weights_tensor, biases_tensor, stride_tensor, padding_tensor, data_type);
    }
    std::cout << "Executing time: " << (exe_time / iter) << " milliseconds"<< std::endl;

    std::cout << "conv layer (1, 7, 512, 512, 3, 1, 1)"<< std::endl;
    batch = 1, in_size = 7, channel = 512, filter = 512, kernel = 3, padding = 1, stride = 1;
    input_tensor = create_tensor(batch, channel, in_size, in_size);
    weights_tensor = create_tensor(filter, channel, kernel, kernel);
    biases_tensor = create_tensor(1, 1, filter, 1);
    stride_tensor = create_tensor(1, 1, stride, stride);
    padding_tensor = create_tensor(0, 0, -padding, -padding);
    exe_time = 0;
    for (int i = 0; i < iter; i++)
    {
        exe_time += run_conv(input_tensor, weights_tensor, biases_tensor, stride_tensor, padding_tensor, data_type);
    }
    std::cout << "Executing time: " << (exe_time / iter) << " milliseconds"<< std::endl;

    return 0;

}

tensor create_tensor(int batch, int feature, int x, int y) {
    tensor ret(batch, feature, x, y);
    return ret;
}
double run_conv(tensor& input_tensor, tensor& weights_tensor, tensor& biases_tensor, tensor& stride_tensor, tensor& padding_tensor, std::string data_type)
{
    //std::cout << "Run conv2d using clDNN." << std::endl;

    // Create an engine
    const bool profiling = true;
    engine engine(profiling);

    // Create input memory for convolution layer
    memory input_prim = memory::allocate(engine, { data_types::f16, format::bfyx, input_tensor });
    memory weights    = memory::allocate(engine, { data_types::f16, format::bfyx, weights_tensor });
    memory biases     = memory::allocate(engine, { data_types::f16, format::bfyx, biases_tensor });
    if (data_type == "float32")
    {
        input_prim = memory::allocate(engine, { data_types::f32, format::bfyx, input_tensor });
        weights    = memory::allocate(engine, { data_types::f32, format::bfyx, weights_tensor });
        biases     = memory::allocate(engine, { data_types::f32, format::bfyx, biases_tensor });
    }
    //std::cout << "memory allocation done." << std::endl;
    set_values(input_prim, get_simple_data<float>(input_prim));
    set_values(weights,    get_simple_data<float>(weights));
    set_values(biases,     get_simple_data<float>(biases));

    // Create a topology with a simple Convolution layer
    topology topology(
        input_layout("conv_input", input_prim.get_layout()),
        data("conv_weights", weights),
        data("conv_biases", biases),
        convolution(
            "conv",
            "conv_input",
            { "conv_weights" },
            { "conv_biases" },
            stride_tensor,
            padding_tensor,
            { 1, 1, 1, 1 },
            false,
            0,
            padding{ { 0, 0, 0, 0 }, 0 })
    );

    build_options build_opt;
    // Optimize_data flag can change weights and outputs layouts. Let take a look at 
    // Set option to optimize data.
    build_opt.set_option(build_option::optimize_data(true));


    network network(engine, topology, build_opt);

    //auto start = std::chrono::steady_clock::now();

    // Set input.
    network.set_input_data("conv_input", input_prim);
    // Ready to go.
    auto outputs = network.execute();
    //auto elapsed =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
    //std::cout << "Elapsed time: " << elapsed.count() << std::endl;
    // Get primitives that were executed and their events needed for profiling
    auto executed_primitives = network.get_executed_primitives();

    // Now, we want to check what is the time of execution of each primitive:
    std::vector<cldnn::instrumentation::profiling_info> profiling_table;
    for (auto& p : executed_primitives)
    {
        profiling_table.push_back({ p.first, p.second.get_profiling_info() });
    }

    // We have table of profiling metrics.
    for (auto& p : profiling_table)
    {
        if (p.name == "conv")
        {
            //std::cout << p.name << ":" << std::endl;
            double executing_time = 0;
            for (auto& q : p.intervals)
            {
                //std::cout << "\t" << q.name << ": " << std::chrono::duration_cast<std::chrono::duration<double, std::chrono::milliseconds::period>>(q.value->value()).count()
                    //<< " milliseconds" << std::endl;
                if (q.name == "executing")
                {
                    executing_time = q.value->value().count() / 1000000.0;
                }
            }
            //std::cout << "\t" << "conv executing time: " << executing_time << " milliseconds"<< std::endl;
            return executing_time;
        }
    }
    return -1000;    
}
