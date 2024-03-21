import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import os

ABS_PATH = os.path.abspath(__file__)
BENCHMARKS_PATH = f"{os.path.abspath(os.path.join(os.path.dirname(ABS_PATH), os.pardir))}/benchmarks"

def trace_cpu_vs_gpu_graph(benchmarks_data):
    # setting number of img as x axis
    x_axis = [test["img_n"] for test in benchmarks_data["cpu"]["cpu_version"]["test_performed"]]
    # calculating avg elapsed times for each test
    y_axis_cpu = [test["elapsed_times"] for test in benchmarks_data["cpu"]["cpu_version"]["test_performed"]]
    y_axis_cpu = [sum(elapsed_times)/len(elapsed_times) for elapsed_times in y_axis_cpu]
    y_axis_gpu = [test["elapsed_times"] for test in benchmarks_data["gpu"]["gpu_version"]["test_performed"]]
    # removing first elements of gpu version because their are always outliers
    for elapsed_times in y_axis_gpu:
        elapsed_times.pop(0)
    y_axis_gpu = [sum(elapsed_times)/len(elapsed_times) for elapsed_times in y_axis_gpu]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(x_axis, y_axis_cpu, label="CPU",)
    ax.plot(x_axis, y_axis_gpu, label="GPU")

    # plot vertical lines
    differences = [(y_cpu / y_gpu) for y_cpu, y_gpu in zip(y_axis_cpu, y_axis_gpu)]

    for x, y_cpu, y_gpu, diff in zip(x_axis, y_axis_cpu, y_axis_gpu, differences):
        ax.plot([x, x], [y_cpu, y_gpu], 'r--')
        ax.text(x + 0.1, (y_cpu + y_gpu) / 2, f"{round(diff, 2)}", color='red')

    custom_legend = Line2D([0], [0], color='red', linestyle='--', linewidth=2, markersize=5)

    ax.set_xlabel('Images analysed')
    ax.set_ylabel('Time [s]')
    ax.set_title(f'CLAHE Benchmark CPU vs GPU')

    # Combine default and custom legends
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(custom_legend)
    labels.append('Speed_up')

    ax.legend(handles=handles, labels=labels)
    fig.savefig(f"{BENCHMARKS_PATH}/benchmark_cpu_vs_gpu.png", dpi=600)

def trace_gpu_shared_mem_graph(benchmarks_data):
    # setting number of img as x axis
    x_axis = [test["img_n"] for test in benchmarks_data["gpu"]["gpu_version"]["test_performed"]]
    # calculating avg elapsed times for each test
    y_axis_gpu = [test["elapsed_times"] for test in benchmarks_data["gpu"]["gpu_version"]["test_performed"]]
    # removing first elements of gpu version because their are always outliers
    for elapsed_times in y_axis_gpu:
        elapsed_times.pop(0)
    y_axis_gpu = [sum(elapsed_times)/len(elapsed_times) for elapsed_times in y_axis_gpu]
    y_axis_gpu_mem_shared = [test["elapsed_times"] for test in benchmarks_data["gpu_mem_shared"]["gpu_mem_shared_version"]["test_performed"]]
    y_axis_gpu_mem_shared = [sum(elapsed_times)/len(elapsed_times) for elapsed_times in y_axis_gpu_mem_shared]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(x_axis, y_axis_gpu, label="GPU")
    ax.plot(x_axis, y_axis_gpu_mem_shared, label="GPU with shared memory")

    ax.set_xlabel('Images analysed')
    ax.set_ylabel('Time [s]')
    ax.set_title(f'CLAHE Benchmark GPU vs GPU with shared memory')
    ax.legend()

    fig.savefig(f"{BENCHMARKS_PATH}/benchmark_gpu_mem_shared.png", dpi=600)


def main():

    # load benchmarks data
    benchmarks_data_graph1 = dict()
    with open(f"{BENCHMARKS_PATH}/benchmark_2/cpu.json",'r') as json_file:
        benchmarks_data_graph1["cpu"] = json.load(json_file)
    with open(f"{BENCHMARKS_PATH}/benchmark_2/gpu.json",'r') as json_file:
        benchmarks_data_graph1["gpu"] = json.load(json_file)
    with open(f"{BENCHMARKS_PATH}/benchmark_2/gpu_mem_shared.json",'r') as json_file:
        benchmarks_data_graph1["gpu_mem_shared"] = json.load(json_file)
    
    #trace graph
    trace_cpu_vs_gpu_graph(benchmarks_data_graph1)

    # load benchmarks data
    benchmarks_data_graph2 = dict()
    with open(f"{BENCHMARKS_PATH}/benchmark_4/gpu.json",'r') as json_file:
        benchmarks_data_graph2["gpu"] = json.load(json_file)
    with open(f"{BENCHMARKS_PATH}/benchmark_4/gpu_mem_shared.json",'r') as json_file:
        benchmarks_data_graph2["gpu_mem_shared"] = json.load(json_file)
    
    #trace graph
    trace_gpu_shared_mem_graph(benchmarks_data_graph2)


if __name__ == "__main__":
    main()