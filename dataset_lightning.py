from dataset import *
import lightning as L
class TennisDataModule(L.LightningDataModule):
    def __init__(self, root, frame_in, is_sequential, train_transform = None, test_transform = None, train_games = [i for i in range(1, 9)], r = 2.5, w = 512, h = 288, batch_size = 2, num_workers = 2, shuffle = True):
        super().__init__()
        self.root = Path(root)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.frame_in = frame_in
        self.is_sequential = is_sequential
        self.games = train_games
        self.r = r
        self.w = w
        self.h = h
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_data = Tennis(root = self.root, train = True, frame_in = self.frame_in, is_sequential = self.is_sequential, transform = self.train_transform, train_games = self.games, r = self.r, w = self.w, h = self.h)
            self.val_data = Tennis(root = self.root, train = False, frame_in = self.frame_in, is_sequential = self.is_sequential, transform = self.test_transform, train_games = self.games, r = self.r, w = self.w, h = self.h)

        if stage == 'test':
            self.test_data = Tennis(root = self.root, train = False, frame_in = self.frame_in, is_sequential = self.is_sequential, transform = self.test_transform, train_games = self.games, r = self.r, w = self.w, h = self.h)

        if stage == 'predict':
            self.predict_data = Tennis(root = self.root, train = False, frame_in = self.frame_in, is_sequential = self.is_sequential, transform = self.test_transform, train_games = self.games, r = self.r, w = self.w, h = self.h)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = self.shuffle, persistent_workers = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False, persistent_workers = True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False, persistent_workers = True)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False, persistent_workers = True)
