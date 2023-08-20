from torch.utils.data import Dataset
import bisect

class JsonDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.filenames = []
        # https://stackoverflow.com/questions/55109684/how-to-handle-large-json-file-in-pytorch#63244085


        