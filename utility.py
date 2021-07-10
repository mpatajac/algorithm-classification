import time

# -------------------------------------------------------
# Base path to these files
# Used to ease the adaption when not running localy
# (e.g. on Google Colab)
base_path = ""


# -------------------------------------------------------
# Time measuring utility (decorator)


def fix_zeros(x): return f"{'0' if len(str(x)) == 1 else ''}{x}"


def display_time(t):
    t = int(t)
    return f"{fix_zeros(t // 60)}:{fix_zeros(t % 60)}"


def measure_time(func):
    def inner(*args, **kwargs):
        start = time.time()

        return_value = func(*args, **kwargs)

        end = time.time()
        print(f"Finished `{func.__name__}` in {display_time(end - start)}.")

        return return_value

    return inner


# -------------------------------------------------------
# Pipe function (execute several functions in a series)

def pipe(data, *args):
    # TODO?: implement using `fold` (reduce)
    for f in args:
        data = f(data)
    return data


# -------------------------------------------------------
# Map function series to an iterable

def pipe_map(iterable, *args):
    return map(
        lambda i: pipe(i, *args),
        iterable
    )

# -------------------------------------------------------
# -------------------------------------------------------
# Tests


def pipe_test():
    data = [1, 2, 3, 4, 5]
    def func1(x): return map(lambda x_i: x_i * 2, x)
    def func2(x): return map(lambda x_i: x_i - 1, x)
    def func3(x): return map(lambda x_i: x_i**2,  x)

    result = pipe(
        data,
        func1,
        func2,
        func3,
        list
    )

    assert(result == [1, 9, 25, 49, 81]), "`Pipe` test failed."


def pipe_map_test():
    data = [1, 2, 3, 4, 5]
    def func1(x): return x * 2
    def func2(x): return x - 1
    def func3(x): return x**2

    result = list(pipe_map(
        data,
        func1,
        func2,
        func3
    ))

    assert(result == [1, 9, 25, 49, 81]), "`Pipe Map` test failed."


if __name__ == "__main__":
    pipe_test()
    pipe_map_test()
