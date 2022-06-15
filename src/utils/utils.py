def print_metrics(epoch, iter, loss):
    f_str = "{: <10} {: <10} {: <10.5}"
    l_str = "-" * 45

    if iter % 300 == 0:
        print(l_str)
        print(f_str.format("Epoch", "Iter", "Loss"))
        print(l_str)
    print(f_str.format(epoch, iter, loss), end="\r" if iter % 50 else "\n")
