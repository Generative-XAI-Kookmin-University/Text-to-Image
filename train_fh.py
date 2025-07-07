import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from Flaw_Highlighter import FlawHighlighter
from tqdm import tqdm
import wandb
import os
import itertools

class Dataset(Dataset):
    def __init__(
            self,
            folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_horizontal_flip=False,
            convert_image_to=None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = T.Compose([
            T.Lambda(lambda x: x.convert('RGB')),  # 모든 이미지를 RGB로 변환
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else T.Lambda(lambda x: x),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class FHTrainer(object):
    def __init__(self, FH, real_img, gen_img, image_size, FH_ckpt=None, batch_size=64, lr=2e-5, adam_betas=(0.5, 0.999), num_epoch=200):
        super().__init__()

        self.FH = FH
        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_epoch = num_epoch

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FH = self.FH.to(self.device)
        print('device:', self.device)

        if FH_ckpt:
            ckpt = torch.load(FH_ckpt)
            ckpt_state_dict = ckpt["model_state_dict"]
            model_state_dict = self.FH.state_dict()

            filtered_state_dict = {}
            for k, v in ckpt_state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"Shape mismatch for {k}: checkpoint {v.shape}, model {model_state_dict[k].shape}")
                else:
                    print(f"Unexpected key in checkpoint: {k}")

            model_state_dict.update(filtered_state_dict)
            self.FH.load_state_dict(model_state_dict, strict=False)
            print('checkpoint accuracy:', ckpt['accuracy'])

        self.criterion = nn.NLLLoss()

        self.real_dataset = Dataset(real_img, self.image_size)
        self.gen_dataset = Dataset(gen_img, self.image_size)

        real_data_size = len(self.real_dataset)
        gen_data_size = len(self.gen_dataset)

        real_test_size = int(real_data_size * 0.1)
        gen_test_size = int(gen_data_size * 0.1)

        self.train_real, self.test_real = random_split(self.real_dataset, [real_data_size - real_test_size, real_test_size])
        self.train_gen, self.test_gen = random_split(self.gen_dataset, [gen_data_size - gen_test_size, gen_test_size])

        self.train_real_dl = DataLoader(self.train_real, batch_size=self.batch_size, shuffle=True)
        self.test_real_dl = DataLoader(self.test_real, batch_size=self.batch_size, shuffle=False)
        self.train_gen_dl = DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True)
        self.test_gen_dl = DataLoader(self.test_gen, batch_size=self.batch_size, shuffle=False)

        self.opt = AdamW(self.FH.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.scheduler = OneCycleLR(self.opt, max_lr=self.lr, epochs=self.num_epoch, steps_per_epoch=len(self.train_real_dl))

        self.best_acc = 0
        
    def train(self):
        wandb.init(project="uvit-fh")

        for epoch in range(1, self.num_epoch + 1):
            self.FH.train()
            epoch_loss = []

            # 더 긴 데이터로더에 맞춰서 처리
            max_len = max(len(self.train_real_dl), len(self.train_gen_dl))
            pbar = tqdm(range(max_len), desc=f"Epoch {epoch}/{self.num_epoch}")
            
            real_iter = itertools.cycle(self.train_real_dl)
            gen_iter = itertools.cycle(self.train_gen_dl)

            for _ in pbar:
                real_images = next(real_iter).to(self.device)
                gen_images = next(gen_iter).to(self.device)

                # 배치 크기 맞추기
                min_batch_size = min(real_images.size(0), gen_images.size(0))
                real_images = real_images[:min_batch_size]
                gen_images = gen_images[:min_batch_size]

                real_labels = torch.ones(min_batch_size, dtype=torch.long, device=self.device)
                fake_labels = torch.zeros(min_batch_size, dtype=torch.long, device=self.device)

                real_outputs = self.FH(real_images)
                fake_outputs = self.FH(gen_images)

                log_real_outputs = torch.log(real_outputs + 1e-9)
                log_fake_outputs = torch.log(fake_outputs + 1e-9)

                real_loss = self.criterion(log_real_outputs, real_labels)
                fake_loss = self.criterion(log_fake_outputs, fake_labels)

                total_loss = (real_loss + fake_loss) / 2

                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
                self.scheduler.step()

                epoch_loss.append(total_loss.item())
                pbar.set_postfix({'loss': total_loss.item()})
                wandb.log({"iteration loss": total_loss})

            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            wandb.log({"epoch loss": avg_epoch_loss})

            print(f"Epoch [{epoch}/{self.num_epoch}], Loss: {avg_epoch_loss}")

            if not os.path.exists('./fh_ckpt/'):
                os.makedirs('./fh_ckpt/')
            
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': self.FH.state_dict(), 
            #     'optimizer_state_dict': self.opt.state_dict(),  
            #     'scheduler_state_dict': self.scheduler.state_dict(),
            #     'loss': avg_epoch_loss 
            # }, f'./fh_ckpt/FH_{epoch}.pth')

            # 평가 단계
            self.FH.eval()
            all_real_preds = []
            all_fake_preds = []
            all_real_labels = []
            all_fake_labels = []

            with torch.no_grad():
                # Real 이미지 평가
                for real_images in self.test_real_dl:
                    real_images = real_images.to(self.device)
                    real_outputs = self.FH(real_images)
                    real_preds = real_outputs.argmax(dim=1).cpu().detach().numpy()
                    real_labels = np.ones(len(real_images))
                    
                    all_real_preds.extend(real_preds)
                    all_real_labels.extend(real_labels)
                
                # Generated 이미지 평가
                for generated_images in self.test_gen_dl:
                    generated_images = generated_images.to(self.device)
                    fake_outputs = self.FH(generated_images)
                    fake_preds = fake_outputs.argmax(dim=1).cpu().detach().numpy()
                    fake_labels = np.zeros(len(generated_images))
                    
                    all_fake_preds.extend(fake_preds)
                    all_fake_labels.extend(fake_labels)
                    
            real_preds = np.array(all_real_preds)
            fake_preds = np.array(all_fake_preds)
            real_labels = np.array(all_real_labels)
            fake_labels = np.array(all_fake_labels)

            real_preds_rounded = np.round(real_preds)
            fake_preds_rounded = np.round(fake_preds)

            real_acc = accuracy_score(real_labels, real_preds_rounded)
            fake_acc = accuracy_score(fake_labels, fake_preds_rounded)
            avg_acc = (real_acc + fake_acc) / 2

            avg_f1 = f1_score(np.concatenate([real_labels, fake_labels]), np.concatenate([real_preds_rounded, fake_preds_rounded]))

            avg_roc_auc = roc_auc_score(np.concatenate([real_labels, fake_labels]), np.concatenate([real_preds, fake_preds]))

            print(f"Real Acc: {real_acc:.4f}, Fake Acc: {fake_acc:.4f}, Avg Acc: {avg_acc:.4f}")
            print(f"F1 Score: {avg_f1:.4f}, ROC AUC: {avg_roc_auc:.4f}")

            wandb.log({
                "epoch": epoch,
                "real_accuracy": real_acc,
                "fake_accuracy": fake_acc,
                "avg_accuracy": avg_acc,
                "f1_score": avg_f1,
                "roc_auc": avg_roc_auc
            })

            if avg_acc > self.best_acc:
                self.best_acc = avg_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.FH.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_epoch_loss,
                    'roc_auc': avg_roc_auc,
                    'accuracy': avg_acc,
                    'f1_score': avg_f1
                }, f'./fh_ckpt/FH_best_{epoch}.pth')
                print(f"New best model saved! Accuracy: {avg_acc:.4f}")
            

if __name__ == '__main__':

    params = {
        'nc' : 3,
        'ndf' : 32,
        }

    FH = FlawHighlighter(params)
    #FH_ckpt = './FH_best_7_share.pth'

    FH_trainer = FHTrainer(FH=FH,
                        real_img='/root/coco/val2017/',
                        gen_img='/root/coco/my_generated_images/',
                        image_size=256)

    FH_trainer.train()