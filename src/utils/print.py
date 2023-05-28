from src.utils.params import count_model_params


def train_start_print(model):
    e_string = "=" * 45

    num_params = count_model_params(model)
    num_params_m = num_params / 1e6

    print("\n" + e_string)
    print("Starting training!")
    print("Num model params: {num_params_m:.3f}M".format(num_params_m=num_params_m))
    print(e_string + "\n")


def iter_print(iter, train_loss, newline_interval=50):
    l_string = "-" * 45
    f_str = "{: <10} {: <10.5}"

    if iter % 500 == 0:
        print(l_string)
        print(f_str.format("Iter", "Train loss"))
        print(l_string)
    print(f_str.format(iter, train_loss), end="\r" if iter % newline_interval else "\n")


def evaluation_print(losses):
    e_string = "=" * 45

    print("\n\n" + e_string)
    print("Evaluation done!")
    print("Mean train loss: {mean_loss:.3f}".format(mean_loss=losses["train"]))
    print("Mean validation loss: {mean_loss:.3f}".format(mean_loss=losses["val"]))
    print(e_string + "\n")
