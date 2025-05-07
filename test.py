import os
import argparse
import torch
import torch.nn.functional as F
import librosa
import time
import numpy as np
from tqdm import tqdm
from emage_utils.motion_io import beat_format_save
from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel

class EmageMotionGenerator:
    def __init__(self, model_path='./', device=None):
        """
        初始化Emage运动生成器
        
        参数:
            model_path: 模型文件路径
            device: 使用的设备(cuda/cpu)，如果为None则自动选择
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self._load_models()
        self.model.eval()
        self.motion_vq.eval()
        
        # 获取配置参数
        self.sr = self.model.cfg.audio_sr
        self.pose_fps = self.model.cfg.pose_fps
    
    def _load_models(self):
        """加载所有必要的模型"""
        face_motion_vq = EmageVQVAEConv.from_pretrained(self.model_path, subfolder="emage_vq/face").to(self.device)
        upper_motion_vq = EmageVQVAEConv.from_pretrained(self.model_path, subfolder="emage_vq/upper").to(self.device)
        lower_motion_vq = EmageVQVAEConv.from_pretrained(self.model_path, subfolder="emage_vq/lower").to(self.device)
        hands_motion_vq = EmageVQVAEConv.from_pretrained(self.model_path, subfolder="emage_vq/hands").to(self.device)
        global_motion_ae = EmageVAEConv.from_pretrained(self.model_path, subfolder="emage_vq/global").to(self.device)
        
        self.motion_vq = EmageVQModel(
            face_model=face_motion_vq, upper_model=upper_motion_vq,
            lower_model=lower_motion_vq, hands_model=hands_motion_vq,
            global_model=global_motion_ae).to(self.device)
        
        self.model = EmageAudioModel.from_pretrained(self.model_path, subfolder="checkpoints").to(self.device)
    
    def generate_motion(self, audio_path, save_folder):
        """
        从音频生成运动数据
        
        参数:
            audio_path: 输入音频文件路径
            save_folder: 结果保存目录
            
        返回:
            生成的动画帧数
        """
        os.makedirs(save_folder, exist_ok=True)
        
        # 加载音频
        audio, _ = librosa.load(audio_path, sr=self.sr)
        audio = torch.from_numpy(audio).to(self.device).unsqueeze(0)
        speaker_id = torch.zeros(1,1).long().to(self.device)
        
        with torch.no_grad():
            trans = torch.zeros(1, 1, 3).to(self.device)
            
            # 推理
            latent_dict = self.model.inference(audio, speaker_id, self.motion_vq, masked_motion=None, mask=None)
            
            # 处理各部位潜在表示
            face_latent = latent_dict["rec_face"] if self.model.cfg.lf > 0 and self.model.cfg.cf == 0 else None
            upper_latent = latent_dict["rec_upper"] if self.model.cfg.lu > 0 and self.model.cfg.cu == 0 else None
            hands_latent = latent_dict["rec_hands"] if self.model.cfg.lh > 0 and self.model.cfg.ch == 0 else None
            lower_latent = latent_dict["rec_lower"] if self.model.cfg.ll > 0 and self.model.cfg.cl == 0 else None
            
            # 处理分类索引
            face_index = torch.max(F.log_softmax(latent_dict["cls_face"], dim=2), dim=2)[1] if self.model.cfg.cf > 0 else None
            upper_index = torch.max(F.log_softmax(latent_dict["cls_upper"], dim=2), dim=2)[1] if self.model.cfg.cu > 0 else None
            hands_index = torch.max(F.log_softmax(latent_dict["cls_hands"], dim=2), dim=2)[1] if self.model.cfg.ch > 0 else None
            lower_index = torch.max(F.log_softmax(latent_dict["cls_lower"], dim=2), dim=2)[1] if self.model.cfg.cl > 0 else None

            # 解码运动
            all_pred = self.motion_vq.decode(
                face_latent=face_latent, upper_latent=upper_latent, lower_latent=lower_latent, hands_latent=hands_latent,
                face_index=face_index, upper_index=upper_index, lower_index=lower_index, hands_index=hands_index,
                get_global_motion=True, ref_trans=trans[:,0])
        
        # 保存结果
        motion_pred = all_pred["motion_axis_angle"]
        t = motion_pred.shape[1]
        motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
        face_pred = all_pred["expression"].cpu().numpy().reshape(t, -1)
        trans_pred = all_pred["trans"].cpu().numpy().reshape(t, -1)
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        npz_path = os.path.join(save_folder, f"{base_name}_output.npz")
        
        beat_format_save(npz_path,
                        motion_pred, upsample=30//self.pose_fps, 
                        expressions=face_pred, trans=trans_pred)
        
        return t

def process_audio_folder(audio_folder="./examples/audio", 
                         save_folder="./examples/motion",
                         model_path='./'):
    """
    处理整个音频文件夹
    
    参数:
        audio_folder: 输入音频文件夹路径
        save_folder: 结果保存目录
        model_path: 模型文件路径
    """
    # 初始化生成器
    generator = EmageMotionGenerator(model_path=model_path)
    
    # 获取音频文件列表
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(".wav")]
    
    # 处理所有音频文件
    all_t = 0
    start_time = time.time()
    
    for audio_path in tqdm(audio_files, desc="Generating motions"):
        all_t += generator.generate_motion(audio_path, save_folder)
    
    # 打印统计信息
    print(f"Generated total {all_t/generator.pose_fps:.2f} seconds motion in {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder", type=str, default="./examples/audio")
    parser.add_argument("--save_folder", type=str, default="./examples/motion")
    parser.add_argument("--model_path", type=str, default="./")
    args = parser.parse_args()
    
    process_audio_folder(
        audio_folder=args.audio_folder,
        save_folder=args.save_folder,
        model_path=args.model_path
    )