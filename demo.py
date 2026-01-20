import os
import random
import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from model import VSSM


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'Checkpoints', 'ERTSegNet.pth')
DATA_ROOT = os.path.join(CURRENT_DIR, 'Mini Dataset')

MODEL_CONFIG = {
    'weights_path': CHECKPOINT_PATH,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'input_size': (256, 1024),
    'dims': [96, 192, 384, 768],          
    'depths': [2, 2, 9, 2],
    'dims_decoder': [768, 384, 192, 96],
    'depths_decoder': [2, 9, 2, 2],
    'patch_size': 4,
    'use_unetpp': True,
    'pp_depth': 2,
    'pp_residual': True
}

class DemoDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.labels_dir = os.path.join(root_dir, 'Labels')
        self.inputs_root = os.path.join(root_dir, 'Inputs')
        
        self.file_names = [f for f in os.listdir(self.labels_dir) if f.endswith('.mat')]
        self.file_names.sort()
        
        if len(self.file_names) == 0:
            print(f"Error: No label files found in {self.labels_dir}")

        self.scales = ['small', 'middle', 'large']

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        label_filename = self.file_names[idx]
        file_id = label_filename.split('_')[-1] 
        input_filename = f"scinv_{file_id}"
        

        selected_scale = np.random.choice(self.scales)
        

        input_path = os.path.join(self.inputs_root, selected_scale, input_filename)
        label_path = os.path.join(self.labels_dir, label_filename)

        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"{input_path} not found.")
                
            mat_x = scio.loadmat(input_path)
            x = mat_x['SC'].astype(np.float32)
            if x.ndim == 2:
                x = x[None, ...] 


            mat_y = scio.loadmat(label_path)
            y = mat_y['label'].astype(np.float32)
            y = (y > 0).astype(np.float32)[None, ...] 

            return torch.from_numpy(x), torch.from_numpy(y), selected_scale

        except Exception as e:
            print(f"Error loading {input_filename} from {selected_scale}: {e}")
            return torch.zeros(1, 256, 1024), torch.zeros(1, 256, 1024), "error"

def calculate_metrics(pred, label, threshold=0.5):
    pred_flat = (pred.flatten() > threshold).astype(int)
    label_flat = (label.flatten() > 0.5).astype(int)

    intersection = (pred_flat & label_flat).sum()
    union = (pred_flat | label_flat).sum()
    iou = intersection / (union + 1e-6)

    f1 = f1_score(label_flat, pred_flat, zero_division=0)
    return iou, f1


def main():
    print(f" Starting ERTSegNet Demo...")
    print(f" Device: {MODEL_CONFIG['device']}")
    print(f" Weights: {MODEL_CONFIG['weights_path']}")
    

    model = VSSM(
        patch_size=MODEL_CONFIG['patch_size'], 
        in_chans=1, 
        num_classes=1, 
        dims=MODEL_CONFIG['dims'], 
        depths=MODEL_CONFIG['depths'],
        dims_decoder=MODEL_CONFIG['dims_decoder'],
        depths_decoder=MODEL_CONFIG['depths_decoder'],
        use_unetpp_local=MODEL_CONFIG['use_unetpp'],
        pp_depth=MODEL_CONFIG['pp_depth'],
        pp_residual=MODEL_CONFIG['pp_residual']
    )
    
    if os.path.exists(MODEL_CONFIG['weights_path']):
        checkpoint = torch.load(MODEL_CONFIG['weights_path'], map_location=MODEL_CONFIG['device'])
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(state_dict, strict=False)
            print(" Model weights loaded successfully.")
        except Exception as e:
            print(f" Weight loading warning: {e}")
    else:
        print(f" Error: Checkpoint not found at {MODEL_CONFIG['weights_path']}")
        return

    model.to(MODEL_CONFIG['device'])
    model.eval()


    if not os.path.exists(DATA_ROOT):
        print(f" Data root '{DATA_ROOT}' not found.")
        return

    dataset = DemoDataset(DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"   Data: Found {len(dataset)} samples in '{DATA_ROOT}'")


    all_labels, all_preds = [], []
    total_iou, total_f1 = 0, 0
    
    print("\nProcessing samples (Scale is randomly selected per sample)...")
    with torch.no_grad():
        for i, (x, y, scale) in enumerate(tqdm(dataloader)):
            if scale[0] == "error": continue

            x = x.to(MODEL_CONFIG['device'])
            

            output = model(x)
            pred = torch.sigmoid(output)
            
            pred_np = pred.cpu().numpy().squeeze()
            y_np = y.cpu().numpy().squeeze()
            

            all_preds.append(pred_np.flatten()[::100]) 
            all_labels.append(y_np.flatten()[::100])
            

            iou, f1 = calculate_metrics(pred_np, y_np)
            total_iou += iou
            total_f1 += f1
            

            if i < 5:
                tqdm.write(f"   Sample {i+1}: Scale={scale[0]:<6} | IoU={iou:.4f} | F1={f1:.4f}")


    avg_iou = total_iou / len(dataset)
    avg_f1 = total_f1 / len(dataset)
    
    print("\n" + "="*40)
    print(f"     Final Results (Avg of {len(dataset)} samples):")
    print(f"   mIoU:      {avg_iou:.4f}")
    print(f"   F1-Score:  {avg_f1:.4f}")
    print("="*40)


    try:
        y_true = np.concatenate(all_labels)
        y_scores = np.concatenate(all_preds)
        
        plt.figure(figsize=(12, 5))
        

        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='#d62728', lw=2, label=f'ERTSegNet (AUC={roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        

        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(recall, precision, color='#1f77b4', lw=2, label=f'ERTSegNet (AP={ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('demo_result.png', dpi=300)
        print(" Plot saved to 'demo_result.png'")
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == '__main__':
    main()