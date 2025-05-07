# copy the ./emage_evaltools folder into your folder
from emage_evaltools.metric import FGD, BC, L1Div, LVDFace, MSEFace

# init
fgd_evaluator = FGD(download_path="./emage_evaltools/")
bc_evaluator = BC(download_path="./emage_evaltools/", sigma=0.3, order=7)
l1div_evaluator= L1div()
lvd_evaluator = LVDFace()
mse_evaluator = MSEFace()

# Example usage
for motion_pred in all_motion_pred:
    # bc and l1 require position representation
    motion_position_pred = get_motion_rep_numpy(motion_pred, device=device, betas=betas)["position"] # t*55*3
    motion_position_pred = motion_position_pred.reshape(t, -1)
    # ignore the start and end 2s, this may for beat dataset only
    audio_beat = bc_evaluator.load_audio(test_file["audio_path"], t_start=2 * 16000, t_end=int((t-60)/30*16000))
    motion_beat = bc_evaluator.load_motion(motion_position_pred, t_start=60, t_end=t-60, pose_fps=30, without_file=True)
    bc_evaluator.compute(audio_beat, motion_beat, length=t-120, pose_fps=30)

    l1_evaluator.compute(motion_position_pred)
    
    face_position_pred = get_motion_rep_numpy(motion_pred, device=device, expressions=expressions_pred, expression_only=True, betas=betas)["vertices"] # t -1
    face_position_gt = get_motion_rep_numpy(motion_gt, device=device, expressions=expressions_gt, expression_only=True, betas=betas)["vertices"]
    lvd_evaluator.compute(face_position_pred, face_position_gt)
    mse_evaluator.compute(face_position_pred, face_position_gt)
    
    # fgd requires rotation 6d representaiton
    motion_gt = torch.from_numpy(motion_gt).to(device).unsqueeze(0)
    motion_pred = torch.from_numpy(motion_pred).to(device).unsqueeze(0)
    motion_gt = rc.axis_angle_to_rotation_6d(motion_gt.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
    motion_pred = rc.axis_angle_to_rotation_6d(motion_pred.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
    fgd_evaluator.update(motion_pred.float(), motion_gt.float())
    
metrics = {}
metrics["fgd"] = fgd_evaluator.compute()
metrics["bc"] = bc_evaluator.avg()
metrics["l1"] = l1_evaluator.avg()
metrics["lvd"] = lvd_evaluator.avg()
metrics["mse"] = mse_evaluator.avg()


#Hyperparameters may vary depending on the dataset.
#For example, for the BEAT dataset, we use (0.3, 7); for the TalkShow dataset, we use (0.5, 7). You may adjust based on your data.