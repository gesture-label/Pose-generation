•test.py: 将音频文件转换为SMPL-X格式的动作数据(.npz) 

• render.py: 将生成的SMPL-X动作数据渲染为视频

步骤1: 生成动作数据


python test.py --audio_path ./examples/audio/your_audio.wav --output_path ./examples/motion/output_motion.npz

参数说明:

--audio_path: 输入音频文件路径 (默认: ./examples/audio/)

--output_path: 输出npz文件路径 (默认: ./examples/motion/)

步骤2: 渲染视频


python render.py --motion_path ./examples/motion/output_motion.npz ./video_path --output_video rendered_video.mp4
