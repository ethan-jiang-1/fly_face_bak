import os
import gc
import platform

if platform == "linux" or platform == "linux2":
    import matplotlib
    if matplotlib.get_backend() != "TkAgg":
        matplotlib.use("Agg")

def _colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}

    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def _log_colorstr(*input):
    cinput = _colorstr(*input)
    print(cinput)

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
        _log_colorstr("bright_magenta", "consumed memory: {:,} ({:,} - {:,}) @{}:".format(mem_after - mem_before, mem_before, mem_after, func.__name__,))
        return result
    return wrapper

s_mem_history = []
def mem_dump(cp_name):
    global s_mem_history
    mem = mem_process_memory()
    s_mem_history.append(mem)
    _log_colorstr("bright_magenta", "{:,}@{}".format(mem, cp_name))

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
