
import os, re, torch,random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from subprocess import CalledProcessError, run
# from funasr import AutoModel
# from transformers import (
#     AutoModelForSpeechSeq2Seq, 
#     AutoProcessor,
#     pipeline,
#     AutoFeatureExtractor,
#     Wav2Vec2BertForSequenceClassification,
#     AutoTokenizer, 
#     AutoModelForSequenceClassification
# )
# import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available()else "cpu"
DATA_DIR = "./data"
seed = 42

datasets = [
    {
        'name': 'EYASE',
        'dir': f'{DATA_DIR}/EYASE',
        'pattern': r'.*?_(?P<emotion>ang|hap|neu|sad)',
        'mapping': {
            'ang': 'angry',
            'hap': 'happy',
            'neu': 'neutral',
            'sad': 'sad'
        }
    },
    {
        'name': 'ShEMO',
        'dir': f'{DATA_DIR}/ShEMO',
        'pattern': r'[MF]\d+(?P<emotion>[AHNSFW])\d+',
        'mapping': {
            'A': 'angry',
            'H': 'happy',
            'N': 'neutral',
            'S': 'sad',
            'F': 'fear',
            'W': 'surprise'
        }
    },
    {
        'name': 'CREMA-D',
        'dir': f'{DATA_DIR}/AudioWAV',
        'pattern': r'\d+_\w+_(?P<emotion>DIS|ANG|HAP|NEU|FEA|SAD)_\w+',
        'mapping': {
            'ANG': 'angry',
            'HAP': 'happy',
            'NEU': 'neutral',
            'FEA': 'fear',
            'SAD': 'sad',
            'DIS': 'disgust'
        }
    },
    {
        'name': 'RAVDESS',
        'dir': '/home/naif/projects/videoEmotionRecognition/data/preprocessed_faces/audio',
        'pattern': r'01-\d{2}-(?P<emotion>\d{2})-\d{2}-\d{2}-\d{2}-\d{2}\.wav',
        'mapping': {
            "01": "neutral",
            "02": "neutral",  # calm is mapped to neutral
            "03": "happy",
            "04": "sad",
            "05": "angry"
        }
    },
    {
        'name': 'URDU',
        'dir': f'{DATA_DIR}/URDU',
        'use_folder': True,
        'mapping': {
            'Angry': 'angry',
            'Happy': 'happy',
            'Neutral': 'neutral',
            'Sad': 'sad'
        }
    },
    {
        'name': 'UJ_ARABIC',
        'dir': f'{DATA_DIR}/Emotional Speech Dataset',
        'use_folder': True,
        'mapping': {
            'Anger': 'angry',
            'Happiness': 'happy',
            'Neutral': 'neutral',
            'Sadness': 'sad'
        }
    },
    {
        'name': 'SaudiEMO',
        'dir': f'{DATA_DIR}/SaudiEmo',
        'complex_structure': True,
        'direct_folders': [
            '8Njah_Mn_Almot',
            '8Mjholat_Alabwen',
            'TikTok',
            'TikTok2',
            'News',
            'News2',
            'News3',
            'khalatt_20',
            'khalatt1_20',
            'khalatt2_20'
        ],
        'wav_subfolders': [ 'react_ang1', 'react_ang0', 'mresel', 'dawwod'],# 'top5',
        'mandoob_structure': {
            'base_dir': 'MANDOOB (Night Courier)',
            'emotions': ['sad', 'hap', 'nat', 'ang']
        },
        'pattern': r'seg_(?P<emotion>\w+?)\.wav$',
        'mapping': {
            'sad': 'sad',
            'hap': 'happy',
            'nat': 'neutral',
            'ang': 'angry'
        }
    },
    {
        'name': 'SUBESCO',
        'dir': f'{DATA_DIR}/SUBESCO',
        'pattern': r'(?P<emotion>ANGRY|DISGUST|FEAR|HAPPY|NEUTRAL|SAD|SURPRISE)_\d+\.wav',
        'mapping': {
            'ANGRY': 'angry',
            'DISGUST': 'disgust',
            'FEAR': 'fear',
            'HAPPY': 'happy',
            'NEUTRAL': 'neutral',
            'SAD': 'sad',
            'SURPRISE': 'surprise'
        }
    }
]
def setup_env():
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("GPU available")
    else:
        print("GPU *NOT* available. Will use CPU (slow)")

    # Seed random generator for repeatibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    collect_and_process_metadata(datasets)
    torch.cuda.empty_cache()

    os.makedirs("checkpoints", exist_ok=True)

    
def normalize_emotion(emotion: str) -> str:
    """
    Normalize emotion labels to standard format
    """
    emotion_mapping = {
        # Original forms
        'angry': 'angry',
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'fear': 'fear',
        'surprise': 'surprise',
        'disgust': 'disgust',
        # Additional forms found in data
        'nat': 'neutral',
        'ang': 'angry',
        'hap': 'happy'
    }
    return emotion_mapping.get(emotion.lower(), emotion)

def collect_and_process_metadata(datasets: list) -> pd.DataFrame:
    """
    Collect emotion metadata from audio files with normalized emotion labels.
    """
    records = []
    total_files = 0
    
    # Count total files for progress bar
    for dataset in datasets:
        if dataset.get('complex_structure'):  # Handle SaudiEMO's special structure
            # Count files in direct folders
            for folder in dataset['direct_folders']:
                folder_path = os.path.join(dataset['dir'], folder)
                total_files += len([f for f in os.listdir(folder_path) if f.endswith('.wav')])
            
            # Count files in wav subfolders
            for folder in dataset['wav_subfolders']:
                folder_path = os.path.join(dataset['dir'], folder, 'wavs')
                total_files += len([f for f in os.listdir(folder_path) if f.endswith('.wav')])
            
            # Count files in mandoob structure
            mandoob_base = os.path.join(dataset['dir'], dataset['mandoob_structure']['base_dir'])
            for emotion in dataset['mandoob_structure']['emotions']:
                emotion_folder_path = os.path.join(mandoob_base, emotion)
                total_files += len([f for f in os.listdir(emotion_folder_path) if f.endswith('.WAV')])
        else:
            for root, _, files in os.walk(dataset['dir']):
                total_files += len([f for f in files if f.endswith('.wav')])
    
    print(f"\nProcessing {total_files} files...")
    pbar = tqdm(total=total_files, desc="Processing files")
    
    for dataset in datasets:
        print(f"\nProcessing {dataset['name']}...")
        
        if dataset.get('complex_structure'):  # Handle SaudiEMO
            # Process direct folders
            for folder in dataset['direct_folders']:
                folder_path = os.path.join(dataset['dir'], folder)
                process_folder(folder_path, dataset, records, pbar)
            
            # Process wav subfolders
            for folder in dataset['wav_subfolders']:
                folder_path = os.path.join(dataset['dir'], folder, 'wavs')
                process_folder(folder_path, dataset, records, pbar)
            
            # Process mandoob structure
            mandoob_base = os.path.join(dataset['dir'], dataset['mandoob_structure']['base_dir'])
            for emotion in dataset['mandoob_structure']['emotions']:
                emotion_folder_path = os.path.join(mandoob_base, emotion)
                for file in os.listdir(emotion_folder_path):
                    if file.endswith('.WAV'):
                        filepath = os.path.join(emotion_folder_path, file)
                        process_file(filepath, normalize_emotion(emotion), dataset, records)
                        pbar.update(1)
        else:
            # Original processing for other datasets
            for root, _, files in os.walk(dataset['dir']):
                for file in files:
                    if not file.endswith('.wav'):
                        continue
                    
                    filepath = os.path.join(root, file)
                    process_regular_file(filepath, dataset, records, pbar)
    
    pbar.close()
    
    # Create DataFrame and save results
    df = pd.DataFrame(records)
    if not df.empty:
        print("\nFiles per dataset:")
        print(df['Dataset'].value_counts())
        print("\nFiles per emotion:")
        print(df['Emotion'].value_counts())

        csv_path = os.path.join(DATA_DIR, 'metadata.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nMetadata saved to: {csv_path}")
    
    return df

def process_folder(folder_path: str, dataset: dict, records: list, pbar: tqdm) -> None:
    """Helper function to process files in a folder"""
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            filepath = os.path.join(folder_path, file)
            match = re.search(dataset['pattern'], file)
            if match:
                emotion_code = match.group('emotion')
                emotion = dataset['mapping'].get(emotion_code)
                if emotion:
                    process_file(filepath, normalize_emotion(emotion), dataset, records)
            pbar.update(1)

def process_file(filepath: str, emotion: str, dataset: dict, records: list) -> None:
    """Helper function to process a single file"""
    try:
        records.append({
            'Dataset': dataset['name'],
            'Path': filepath,
            'Emotion': emotion
        })
    except Exception as e:
        print(f"\nError processing {filepath}: {e}")

def process_regular_file(filepath: str, dataset: dict, records: list, pbar: tqdm) -> None:
    """Helper function to process files from regular dataset structure"""
    if dataset.get('use_folder', False):
        emotion_code = os.path.basename(os.path.dirname(filepath))
    else:
        match = re.search(dataset['pattern'], os.path.basename(filepath))
        if not match:
            print(filepath)
            pbar.update(1)
            return
        emotion_code = match.group('emotion')
    
    emotion = dataset['mapping'].get(emotion_code)
    if emotion:
        process_file(filepath, normalize_emotion(emotion), dataset, records)
    pbar.update(1)




def train_test_split(X, y=None, test_size=0.25, stratify=None, random_state=None):
    """
    Split DataFrames into random train and test subsets.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features DataFrame to split
    y : pandas.Series or pandas.DataFrame, optional (default=None)
        Target variable to split
    test_size : float, optional (default=0.25)
        Proportion of the dataset to include in the test split (0.0 to 1.0)
    stratify : array-like, optional (default=None)
        If not None, data is split in a stratified fashion using this as class labels
    random_state : int, optional (default=None)
        Controls the shuffling applied to the data before applying the split
    
    Returns:
    --------
    splitting : tuple of pandas.DataFrame
        Returns (train_df, test_df) or (X_train, X_test, y_train, y_test)
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    
    if y is not None and not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series or DataFrame")
        
    if not 0.0 <= test_size <= 1.0:
        raise ValueError("test_size must be between 0 and 1")
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    # Ensure index are reset
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    # Get number of samples
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Create index array
    indices = np.arange(n_samples)
    
    if stratify is not None:
        # Convert stratify to numpy array if it's a pandas Series/DataFrame
        if isinstance(stratify, (pd.Series, pd.DataFrame)):
            stratify_array = stratify.values
        else:
            stratify_array = np.array(stratify)
            
        # Perform stratified split
        classes, y_indices = np.unique(stratify_array, return_inverse=True)
        n_classes = len(classes)
        
        # Calculate samples per class for test set
        class_counts = np.bincount(y_indices)
        test_counts = np.maximum(1, (test_size * class_counts).astype(int))
        
        # Create test indices
        test_indices = []
        for i in range(n_classes):
            class_indices = indices[y_indices == i]
            np.random.shuffle(class_indices)
            test_indices.extend(class_indices[:test_counts[i]])
        
        test_indices = np.array(test_indices)
        train_indices = np.setdiff1d(indices, test_indices)
        
    else:
        # Perform random split
        np.random.shuffle(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    # Split X while preserving DataFrame type and index
    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    
    if y is not None:
        # Split y if provided
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_indices].reset_index(drop=True)
            y_test = y.iloc[test_indices].reset_index(drop=True)
        else:  # DataFrame
            y_train = y.iloc[train_indices].reset_index(drop=True)
            y_test = y.iloc[test_indices].reset_index(drop=True)
        return X_train, X_test, y_train, y_test
    
    return X_train, X_test



def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    if group_name.lower() == "loss":
        ax.set_ylim([None, None])


def replace_insatance(model,replaced, new):
    for name,module in model.named_children():
        if isinstance(module, replaced):
            setattr(model,name,new)
        else:
            replace_insatance(module,replaced, new)


def load_audio(file: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def plot_confusion_matrix(pred, truth):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)

    fig, sub = plt.subplots(figsize=(14, 12))
    with sns.plotting_context("notebook"):
        idx = (confusion_matrix == 0)
        confusion_matrix[idx] = np.nan
        sns.heatmap(confusion_matrix, annot=True, ax=sub, linewidths=0.5, linecolor='lightgray', cbar=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
    
    
def load_model(model_name, model, loading_order:list= ['best', 'last'], checkpoints_dir='checkpoints',):

    try:
        checkpoint = torch.load(f'{checkpoints_dir}/{loading_order[0]}_{model_name}.pt')
    except FileNotFoundError as e:
        print(e)
        try:
            checkpoint = torch.load(f'{checkpoints_dir}/{loading_order[1]}_{model_name}.pt')
        except FileNotFoundError:
            print("New wheigts are initilaized!!")
            return 1

    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Previously trained model weights state_dict loaded...')
    epochs = checkpoint['epochs']
    print(f"Previously trained for {epochs} number of epochs...")

    return epochs +1

