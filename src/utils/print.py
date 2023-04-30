def iter_print(epoch, iter, train_loss):
    l_string = "-" * 45
    f_str = "{: <10} {: <10} {: <10.5}"

    if iter % 500 == 0:
        print(l_string)
        print(f_str.format("Epoch", "Iter", "Loss"))
        print(l_string)
    print(f_str.format(epoch, iter, train_loss), end="\r" if iter % 50 else "\n")


def epoch_print(epoch, val_losses):
    e_string = "=" * 45

    num_val = len(val_losses)
    val_sum = sum(val_losses)
    val_mean = val_sum / num_val

    print("\n\n" + e_string)
    print("Epoch {epoch} done!".format(epoch=epoch))
    print("Total validation loss: {val_sum:.3f}".format(val_sum=val_sum))
    print("Average validation loss: {val_mean:.3f}".format(val_mean=val_mean))
    print(e_string + "\n")
