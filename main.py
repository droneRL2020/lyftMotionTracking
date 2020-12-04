import fire
from trainer import Trainer
from inferencer import Inferencer

def main(TYPE="train", W_PATH=None):
    print("TYPE", TYPE)
    print("W_PATH", W_PATH)
    engine = Trainer(W_PATH) if TYPE == "train" else Inferencer(W_PATH)
    engine.run()

if __name__ == '__main__':
    fire.Fire(main)