import torch
import numpy as np
from scipy import stats
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm.auto import tqdm
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


class EvalDataset(Dataset):
    def __init__(self, image_list_file, image_root, image_size):
        super().__init__()
        self.image_root = image_root
        self.image_size = image_size

        with open(image_list_file, 'r', encoding='utf-8') as f:
            self.paths = [line.strip() for line in f if line.strip()]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        return self.transform(img)


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda()
    model.eval()
    return model, global_step


class StableDiffusionEvaluator:
    def __init__(self,
                 config_path,
                 model_path,
                 image_root,
                 image_list_file,
                 prompts_file,
                 batch_size=10,
                 num_samples=50000,
                 image_size=64,
                 ddim_steps=50,
                 ddim_eta=1.0,
                 device=None,
                 save_generated_images=False,
                 samples_dir='./generated_samples'):
        self.config_path = config_path
        self.model_path = model_path
        self.image_root = image_root
        self.image_list_file = image_list_file
        self.prompts_file = prompts_file
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.image_size = image_size
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_generated_images = save_generated_images
        self.samples_dir = samples_dir
        if self.save_generated_images:
            os.makedirs(self.samples_dir, exist_ok=True)

        self.config = OmegaConf.load(self.config_path)
        self.model, self.global_step = load_model_from_config(self.config, self.model_path)

        self.prompts = self._load_prompts(self.prompts_file)

        self.inception_model = InceptionV3([
            InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        ]).to(self.device)

        self.real_features = None
        self.load_real_features()

    def _load_prompts(self, prompts_file):
        prompts = []
        with open(prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
        print(f"Loaded {len(prompts)} prompts from {prompts_file}")
        return prompts

    def get_inception_features(self, images):
        images = (images + 1) / 2
        self.inception_model.eval()
        with torch.no_grad():
            features = self.inception_model(images)[0]
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_real_features(self):
        dataset = EvalDataset(
            image_list_file=self.image_list_file,
            image_root=self.image_root,
            image_size=self.image_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        n_samples = min(self.num_samples, len(dataset))
        n_batches = int(np.ceil(n_samples / self.batch_size))
        features_list = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Processing real images")):
                if i >= n_batches:
                    break
                batch = batch.to(self.device)
                batch = 2 * batch - 1
                feats = self.get_inception_features(batch)
                features_list.append(feats.cpu())
        self.real_features = torch.cat(features_list, dim=0)[:n_samples]
        print(f"Processed {self.real_features.shape[0]} real images")

    def calculate_inception_score(self, features, splits=10):
        scores = []
        subset = features.shape[0] // splits
        for k in range(splits):
            part = features[k*subset:(k+1)*subset]
            prob = torch.nn.functional.softmax(part, dim=1).cpu().numpy()
            p_y = np.mean(prob, axis=0)
            kl = prob * (np.log(prob+1e-10) - np.log(p_y+1e-10))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))

    def calculate_fid(self, real_features, fake_features):
        mu1, sigma1 = np.mean(fake_features.cpu().numpy(), axis=0), np.cov(fake_features.cpu().numpy(), rowvar=False)
        mu2, sigma2 = np.mean(real_features.cpu().numpy(), axis=0), np.cov(real_features.cpu().numpy(), rowvar=False)
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    def calculate_precision_recall(self, real_feats, fake_feats, k=3):
        real = real_feats.cpu(); fake = fake_feats.cpu()
        d_rr = torch.cdist(real, real).fill_diagonal_(float('inf'))
        tau = d_rr.kthvalue(k, dim=1)[0].median().item()
        d_fr = torch.cdist(fake, real)
        precision = (d_fr.min(dim=1)[0] < tau).float().mean().item()
        d_rf = torch.cdist(real, fake)
        recall = (d_rf.min(dim=1)[0] < tau).float().mean().item()
        return precision, recall

    def compute_generated_statistics(self):
        from ldm.models.diffusion.ddim import DDIMSampler
        sampler = DDIMSampler(self.model)
        total = min(self.num_samples, len(self.prompts))
        rounds = int(np.ceil(total / self.batch_size))
        fake_feats = []
        with torch.no_grad():
            for i in tqdm(range(rounds), desc="Generating samples"):
                start, end = i*self.batch_size, min((i+1)*self.batch_size, total)
                bsize = end - start
                if bsize <= 0: break
                batch_prompts = self.prompts[start:end]
                c = self.model.cond_stage_model(batch_prompts)
                shape = [bsize,
                         self.model.model.diffusion_model.in_channels,
                         self.model.model.diffusion_model.image_size,
                         self.model.model.diffusion_model.image_size]
                samples, _ = sampler.sample(
                    S=self.ddim_steps,
                    conditioning=c,
                    batch_size=bsize,
                    shape=shape[1:],
                    eta=self.ddim_eta,
                    verbose=False
                )
                x_samples = self.model.decode_first_stage(samples)
                if self.save_generated_images:
                    for j in range(bsize):
                        idx = start + j
                        if idx < 100:
                            img = x_samples[j]
                            img = (img+1)/2
                            arr = (img.permute(1,2,0).cpu().numpy()*255).astype('uint8')
                            Image.fromarray(arr).save(os.path.join(self.samples_dir, f'sample_{idx:05d}.png'))
                feats = self.get_inception_features(x_samples)
                fake_feats.append(feats.cpu())
                del samples, x_samples, feats, c; torch.cuda.empty_cache()
        return torch.cat(fake_feats, dim=0)[:total]

    def save_results(self, results):
        out_dir = './eval_results'
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            'model_path': self.model_path,
            'config_path': self.config_path,
            'global_step': self.global_step,
            'num_samples': self.num_samples,
            'batch_size': self.batch_size,
            'ddim_steps': self.ddim_steps,
            'ddim_eta': self.ddim_eta,
            'inception_score': float(results['inception_score']),
            'fid_score': float(results['fid_score']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'timestamp': ts
        }
        fname = os.path.join(out_dir, f'eval_results_{self.global_step}_{ts}.json')
        with open(fname, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Results saved to {fname}")
        if self.save_generated_images:
            print(f"Sample images saved to {self.samples_dir}")

    def evaluate(self):
        print(f"Starting evaluation for global step {self.global_step}")
        fake_feats = self.compute_generated_statistics()
        is_score = self.calculate_inception_score(fake_feats)
        print(f"Inception Score: {is_score:.3f}")
        fid_score = self.calculate_fid(self.real_features, fake_feats)
        print(f"FID Score: {fid_score:.3f}")
        precision, recall = self.calculate_precision_recall(self.real_features, fake_feats)
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        results = {
            'inception_score': is_score,
            'fid_score': fid_score,
            'precision': precision,
            'recall': recall
        }
        self.save_results(results)
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Text-to-Image Stable Diffusion model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the model config (YAML) file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint (.ckpt)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Root directory of real images for FID calculation')
    parser.add_argument('--image_list_file', type=str, required=True,
                        help='Text file listing real image filenames in exact order')
    parser.add_argument('--prompts_file', type=str, required=True,
                        help='Text file containing one prompt/caption per line')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=30000,
                        help='Number of samples to generate')
    parser.add_argument('--ddim_steps', type=int, default=250,
                        help='Number of DDIM sampling steps')
    parser.add_argument('--ddim_eta', type=float, default=1.0,
                        help='DDIM eta parameter (0.0 for deterministic)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save generated images')
    parser.add_argument('--samples_dir', type=str, default='./generated_samples',
                        help='Directory to save generated samples')
    args = parser.parse_args()

    evaluator = StableDiffusionEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        image_root=args.dataset_path,
        image_list_file=args.image_list_file,
        prompts_file=args.prompts_file,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        image_size=256,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        device=None,
        save_generated_images=args.save_images,
        samples_dir=args.samples_dir
    )
    evaluator.evaluate()


if __name__ == '__main__':
    main()
