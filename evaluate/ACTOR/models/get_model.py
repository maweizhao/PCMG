import importlib

JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]

LOSSES = ["rc", "kl", "rcxyz"]  # not used: "hp", "mmd", "vel", "velxyz"

MODELTYPES = ["cvae"]  # not used: "cae"
ARCHINAMES = ["fc", "gru", "transformer", "transgru", "grutrans", "autotrans"]


def get_model(parameters):
    # modeltype:cvae
    modeltype = parameters["modeltype"]
    # archiname:transformer
    archiname = parameters["archiname"]

    # 在src/models中的architectures/transformer
    archi_module = importlib.import_module(f'.architectures.{archiname}', package="evaluate.ACTOR.models")
    # .upper表示大写
    # f-string在形式上是以 f 或者 F 修饰符引领的字符串（f'xxx' 或 F'xxx'），以大括号 {} 标明被替代的字段。
    # f-string本质上不是字符串产常量，而是一个在运行时运算求值的表达式。
    # Encode为transformer.py下的Encoder_TRANSFORMER
    Encoder = archi_module.__getattribute__(f"Encoder_{archiname.upper()}")
    # Decode为transformer.py下的Decoder_TRANSFORMER
    Decoder = archi_module.__getattribute__(f"Decoder_{archiname.upper()}")

    # 在src/models中的modeltype/cvae
    model_module = importlib.import_module(f'.modeltype.{modeltype}', package="evaluate.ACTOR.models")
    # cvae.py中的CVAE
    Model = model_module.__getattribute__(f"{modeltype.upper()}")

    encoder = Encoder(**parameters)
    decoder = Decoder(**parameters)
    
    # lambdas是否包含rcxyz，如果有则为True
    parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
    return Model(encoder, decoder, **parameters).to(parameters["device"])
