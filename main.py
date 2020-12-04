import fire
from trainer import Trainer
# from inferencer

def main(TYPE="train", W_PATH="None"):
    engine = Trainer(W_PATH) if TYPE == "train" else Inferencer(W_PATH)
    engine.run()

if __name__ == '__main__':
    fire.Fire(main)