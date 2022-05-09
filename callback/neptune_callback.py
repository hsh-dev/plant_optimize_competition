
class NeptuneCallback():
    def __init__ (self, callback):
        self.neptune = callback
        
    def save(self, logs):
        if logs.get("train_mae") is not None:
            self.neptune["train/MAE"].log(logs["train_mae"])
            
        if logs.get("train_score") is not None:
            self.neptune["train/Score"].log(logs["train_score"])

        if logs.get("valid_mae") is not None:
            self.neptune["valid/MAE"].log(logs["valid_mae"])
            
        if logs.get("valid_score") is not None:
            self.neptune["valid/Score"].log(logs["valid_score"])
        
        if logs.get("learning_rate") is not None:
            self.neptune["train/LR"].log(logs["learning_rate"])