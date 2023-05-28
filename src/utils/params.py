def count_model_params(model):
    total_params = 0
    for params in list(model.parameters()):
        num = 1
        for size in list(params.size()):
            num = num * size
        total_params += num
    return total_params
