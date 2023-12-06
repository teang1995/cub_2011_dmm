import os
import random
import torch
import torchvision

import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from torchvision import transforms
from torch.utils.data import DataLoader

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        
    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            # import pdb; pdb.set_trace()
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
class TripletCub2011(Cub2011):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        
    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            # import pdb; pdb.set_trace()
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor_sample = self.data.iloc[idx]
        anchor_path = os.path.join(self.root, self.base_folder, anchor_sample.filepath)
        anchor_target = anchor_sample.target - 1  # Targets start at 1 by default, so shift to 0
        anchor_img = self.loader(anchor_path)

        # Select positive example (image with the same class as anchor)
        positive_data = self.data[self.data['target'] == anchor_sample.target]
        positive_sample = positive_data.sample(n=1).iloc[0]
        positive_path = os.path.join(self.root, self.base_folder, positive_sample.filepath)
        positive_target = positive_sample.target - 1
        positive_img = self.loader(positive_path)

        # Select negative example (image with a different class than anchor)
        negative_data = self.data[self.data['target'] != anchor_sample.target]
        negative_sample = negative_data.sample(n=1).iloc[0]
        negative_path = os.path.join(self.root, self.base_folder, negative_sample.filepath)
        negative_target = negative_sample.target - 1
        negative_img = self.loader(negative_path)

        # Apply transformations if specified
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, anchor_target, positive_target, negative_target
    
def recall(embeddings, labels, K = 1):
    prod = torch.mm(embeddings, embeddings.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    D = norm + norm.t() - 2 * prod
    knn_inds = D.topk(1 + K, dim = 1, largest = False)[1][:, 1:]
    return (labels.unsqueeze(-1).expand_as(knn_inds) == labels[knn_inds.flatten()].view_as(knn_inds)).max(1)[0].float().mean()

if __name__ == '__main__':
    random_seed = 1
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torchvision.models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 256)
    model = model.to(device)
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
    train_dataset = TripletCub2011(root='./', train=True, transform=transform, loader=default_loader, download=False)
    test_dataset = Cub2011(root='./', train=False, transform=transform, loader=default_loader, download=False)
    margin = 1.0
    init_lr = 0.05
    criterion = torch.nn.TripletMarginLoss(margin=margin,
                                      p=2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
    num_epochs = 2000
    KS = [1, 2, 4, 8]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    for epoch in range(num_epochs):
        # TRAIN
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            anchor_img, positive_img, negative_img, anchor_target, positive_target, negative_target = batch
            anchor_img, positive_img, negative_img = anchor_img.to(device), positive_img.to(device), negative_img.to(device)
            anchor_embedding, positive_embedding, negative_embedding = model(anchor_img), model(positive_img), model(negative_img)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

        # TEST
        model.eval()
        embeddings_all, labels_all = [], []

        for batch_idx, batch in enumerate(test_loader):
            images, labels = [tensor.cuda() for tensor in batch]
            with torch.no_grad():
                output = model(images)

            embeddings_all.append(output.data.cpu())

            labels_all.append(labels.data.cpu())
        recalls = [recall(torch.cat(embeddings_all), torch.cat(labels_all), K=k) for k in KS]
        for k, recallk in zip(KS, recalls):
            print(f'recall@{k} epoch {epoch}: {recallk}\n')
    import pdb; pdb.set_trace()
    