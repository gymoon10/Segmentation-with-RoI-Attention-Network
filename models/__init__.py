
from models.ERFNet_Semantic_Original import ERFNet_Semantic_Original
from models.ERFNet_Semantic_RoI import ERFNet_Semantic_RoI


def get_model(name, model_opts):
    if name == "ERFNet_Semantic_Original":
        model = ERFNet_Semantic_Original(**model_opts)
        return model

    elif name == "ERFNet_Semantic_RoI":
        model = ERFNet_Semantic_RoI2(**model_opts)
        return model

    else:
        raise RuntimeError("model \"{}\" not available".format(name))
