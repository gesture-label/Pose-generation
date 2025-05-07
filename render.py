# render a npz file to a mesh video
from emage_utils import fast_render
#fast_render.render_one_sequence_no_gt("./examples/motion/2_scott_0_103_103_28s_output.npz", "./examples/result_video/","./examples/audio/2_scott_0_103_103_28s.wav") #predict
fast_render.render_one_sequence_no_gt("/data/gn_2025/Pose_generation/examples/motion/test_output.npz", "./examples/result_video/","/data/gn_2025/Pose_generation/examples/audio/test.wav") #GT