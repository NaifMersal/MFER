import os
import shutil
import glob
import subprocess
# from moviepy.editor import AudioFileClip
from pathlib import Path
import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from PIL import Image
# Let's see if we have an available GPU
import numpy as np
import random


# Seed random generator for repeatibility  
seed = 42
device= "cuda" if torch.cuda.is_available() else "cpu"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from facenet_pytorch import MTCNN
import torchvision.transforms as VT
from PIL import Image

def setup_env():
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("GPU available")
    else:
        print("GPU *NOT* available. Will use CPU (slow)")

    # Download data if not present already
    # if not os.path.exists("data/metadata.csv"):
        # prepare_data()
    df_ravdess = prepare_data(data_root="data/archive (1)/RAVDESS dataset")
    df_ravdess['dataset'] = 'RAVDESS'

    # Process new dataset
    df_new = prepare_data(data_root="data/UJ_ACTORS", label_pos = -1)
    df_new['dataset'] = 'UJ_ACTORS'

    # Merge the two DataFrames
    df_merged = pd.concat([df_ravdess, df_new], ignore_index=True)
    df_merged.to_csv("data/metadata.csv", index=False)

    # compute_mean_and_std()
    # Make checkpoints subdir if not existing
    os.makedirs("checkpoints", exist_ok=True)


# def prapare_data():
#     data_folder = get_data_location()
#     for em in emotions.values():
#         os.makedirs(data_folder + "/" + em, exist_ok=True)

#     # Using glob.glob to get a list of files that match a pattern
#     for path in glob.glob('**/01-*.mp4', recursive=True):
#         vid_dir=data_folder+"/"+emotions[path[-18:-16]]+path[-25:-4]
#         os.makedirs(vid_dir, exist_ok=True)
#         subprocess.call(["ffmpeg", "-i", path,"-vf",f'fps={fps}, scale=256:256', f"{vid_dir}/image_%04d.jpg"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
#         subprocess.call(["ffmpeg", "-y", "-i", path, f"{vid_dir}/audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# def prepare_data():
#     data_folder = get_data_location()


#     df = pd.DataFrame(columns=["video_dir","image_path", "class"])
#     # Using glob.glob to get a list of files that match a pattern
#     files=glob.glob('**/01-*.mp4', recursive=True)
#     for path in tqdm(files,total=len(files),desc="preparing data", ncols=80):
#         vid_dir=data_folder+"/Classes/"+emotions[path[-18:-16]]+path[-25:-4]
#         images_path=f"{vid_dir}/images"
#         os.makedirs(images_path, exist_ok=True)
#         subprocess.call(["ffmpeg", "-y", "-i", path, f"{vid_dir}/raw_audio.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
#         subprocess.call(["ffmpeg", "-i", path,"-vf",f'fps={1/every_s}, scale=256:256', f"{images_path}/%03d.jpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
#         for images in os.listdir(images_path):
#             df.loc[len(df)] = [vid_dir, images,  emotions[path[-18:-16]]]
    
#     df.to_csv(data_folder+"/metadata.csv", index=False)





def prepare_data(
    data_root: str="data/archive (1)/RAVDESS dataset",
    output_root: str = "data/preprocessed_faces",
    fps: int = 5,
    label_pos: int  = 2,
    face_size: int = 224,
    scale_factor: float = 1.3,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Prepare dataset by extracting faces and audio from videos.
    
    Args:
        data_root: Path to dataset root
        output_root: Path to save processed data
        fps: Frames per second to extract
        face_size: Size of face crops
        scale_factor: Scale factor for face margin
        device: Device for MTCNN
    """
    # Initialize MTCNN for face detection
    detector = MTCNN(
        image_size=face_size,
        margin=int(face_size * (scale_factor - 1) / 2),
        device=device,
        select_largest=True,
        post_process=False,
        keep_all=False
    )
    
    # Create output directories
    output_root = Path(output_root)
    faces_dir = output_root / "faces"
    audio_dir = output_root / "audio"
    temp_frames_dir = output_root / "temp_frames"
    faces_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Define emotion mapping
    emotions = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }

    normalize_dict = {
        "natural" : "neutral",
        "fear" : "fearful",
        "disgusted" : "disgust",
    }

    # Initialize DataFrame
    data = []
    
    # Process all videos
    video_paths = list(Path(data_root).rglob("*.*"))
    for video_path in tqdm(video_paths, desc="Processing videos"):
        try:
            # Determine video format and extract relevant information
            if video_path.suffix == ".mp4":
                video_id = video_path.stem
                emotion= video_id.split('-')[label_pos].lower()
            elif video_path.suffix == ".MOV":
                video_id = video_path.stem
                emotion = video_path.stem.split('_')[label_pos].lower()

            else:
                continue
            if emotion in normalize_dict:
                emotion = normalize_dict[emotion]
            if emotion in emotions:
                emotion = emotions[emotion]
            if not (emotion in emotions or emotion in emotions.values()):
                print(emotion)
                continue
                
            # directory video
            video_faces_dir = faces_dir / video_id

            

            
            # Extract audio
            audio_path = audio_dir / f"{video_id}.wav"
            if not os.path.exists(audio_path):
                subprocess.call(
                    ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", str(audio_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
            if not os.path.exists(video_faces_dir):
                # Create temporary directory for frame extraction
                temp_video_frames = temp_frames_dir / video_id
                temp_video_frames.mkdir(exist_ok=True)
                video_faces_dir.mkdir(exist_ok=True)
                # Extract frames using ffmpeg
                subprocess.call(
                    ["ffmpeg", "-i", str(video_path), 
                    "-vf", f"fps={fps},scale=256:256", 
                    str(temp_video_frames / "%03d.jpeg")],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                
                # Process extracted frames
                frame_paths = []
                for frame_path in sorted(temp_video_frames.glob("*.jpeg")):
                    try:
                        # Read image
                        img = Image.open(frame_path)
                        
                        # Detect and crop face
                        face_img = detector(img)
                        
                        if face_img is not None:
                            output_path = video_faces_dir / f"frame_{len(frame_paths):04d}.png"
                            if isinstance(face_img, torch.Tensor):
                                face_img = VT.ToPILImage()(face_img.to(torch.uint8))
                            face_img.save(output_path)
                            frame_paths.append(output_path)
                            
                    except Exception as e:
                        print(f"Error processing frame {frame_path}: {str(e)}")
                        continue
                
                # Clean up temporary frames
                for frame_path in temp_video_frames.glob("*.jpeg"):
                    frame_path.unlink()
                temp_video_frames.rmdir()
                

            data.append({
                'video_id': video_id,
                'frame_dir': video_faces_dir,
                'audio_path': audio_path,
                'emotion': emotion ,
            })
                
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue
    
    # Remove temporary directory if empty
    try:
        temp_frames_dir.rmdir()
    except:
        pass
    
    # Create and save DataFrame
    df = pd.DataFrame(data)
    df['emotion'] = pd.Categorical(df['emotion'])
    df['target'] = df['emotion'].cat.codes
    
    return df
    


def get_data_location():
    """
    Find the location of the dataset
    """
    

    if os.path.exists("data"):
        data_folder = "data"



    else:
        raise IOError("Please download the dataset first")

    return data_folder








# Compute image normalization
def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    # cache_file = "mean_and_std.pt"
    # if os.path.exists(cache_file):
    #     print(f"Reusing cached mean and std")
    #     d = torch.load(cache_file)

    #     return d["mean"], d["std"]

    # folder = get_data_location()
    # ds = datasets.ImageFolder(
    #     folder, transform=transforms.Compose([transforms.ToTensor()])
    # )
    # dl = torch.utils.data.DataLoader(
    #     ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    # )

    # mean = 0.0
    # for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
    #     batch_samples = images.size(0)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    # mean = mean / len(dl.dataset)

    # var = 0.0
    # npix = 0
    # for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
    #     batch_samples = images.size(0)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    #     npix += images.nelement()

    # std = torch.sqrt(var / (npix / 3))

    # # Cache results so we don't need to redo the computation
    # torch.save({"mean": mean, "std": std}, cache_file)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    return mean, std



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
    if isinstance(X,list):
        X = pd.Series(X)
        
    if not isinstance(X, (pd.Series, pd.DataFrame)):
        raise TypeError("X must be a pandas DataFrame")
    
    if y is not None and not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series or DataFrame")
        
    if not 0.0 <= test_size <= 1.0:
        raise ValueError("test_size must be between 0 and 1")
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    

    # ensure the index are reset
    X = X.reset_index(drop=True)
    # Get number of samples
    n_samples = len(X)
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
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
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

def load_model(model_name, model,checkpoints_dir='checkpoints', loading_order = ['best', 'last']):

    try:
        checkpoint = torch.load(f'{checkpoints_dir}/{loading_order[0]}_{model_name}.pt')
    except FileNotFoundError:
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
