
class NeptuneCallback():
    def __init__ (self, callback):
        self.neptune = callback
        
    def save(self, logs):
        if logs.get("train_mse") is not None:
            self.neptune["train/MSE"].log(logs["train_mse"])
            
        if logs.get("train_score") is not None:
            self.neptune["train/Score"].log(logs["train_score"])

        if logs.get("valid_mse") is not None:
            self.neptune["valid/MSE"].log(logs["valid_mse"])
            
        if logs.get("valid_score") is not None:
            self.neptune["valid/Score"].log(logs["valid_score"])