import os
import gc
import platform

if platform == "linux" or platform == "linux2":
    import matplotlib
    if matplotlib.get_backend() != "TkAgg":
        matplotlib.use("Agg")

# inner psutil function
def mem_process_memory():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss
    except:
        return 0
 
# decorator function
def mem_profile(func):
    def wrapper(*args, **kwargs):
 
        mem_before = mem_process_memory()
        result = func(*args, **kwargs)
        mem_after = mem_process_memory()
        print("consumed memory: {:,} ({:,} - {:,}) @{}:".format(mem_after - mem_before, mem_before, mem_after, func.__name__,))
        return result
    return wrapper

s_mem_history = []
def mem_dump(cp_name):
    global s_mem_history
    mem = mem_process_memory()
    s_mem_history.append(mem)
    print(cp_name, "{:,}".format(mem))

def get_mem_history():
    global s_mem_history
    return s_mem_history

def plot_mem_history(title=None):
    import matplotlib.pyplot as plt
    import numpy as np

    np_mh = np.array(s_mem_history).astype("float")
    np_mh /= float(1024 * 1024)

    plt.figure(figsize=(8, 8))
    if title is not None:
        plt.title(title)
    plt.plot([i for i in range(len(np_mh))], np_mh)
    plt.ylabel('mem(G)')
    plt.show()


if __name__ == '__main__':
    @mem_profile
    def _test_func():
        x = [1] * (10 ** 7)  # noqa: F841
        y = [2] * (4 * 10 ** 8)  # noqa: F841
        mem_dump("ck21")
        del x
        mem_dump("ck22")
        #gc.collect()
        return y

    @mem_profile
    def _test_caller():
        y = _test_func() # noqa: F841
        mem_dump("ck13")
        #print(type(y))
        del y
        mem_dump("ck14")
        #gc.collect()

    @mem_profile
    def _test_app():
        _test_caller()
        #gc.collect() 

    gc.collect()
    mem_dump("ck00")
    y = _test_app()
    mem_dump("ck01")
    gc.collect()
    mem_dump("ck03")
