import numpy as np

def extend_demo(base_demo, max_steps_per_episode:int, extension_type:str, demo_extend_num=None, time_feat=False):
    if len(base_demo) >= max_steps_per_episode:
        demo = base_demo
    elif extension_type == "extend_last":
        demo = base_demo
        # append last state as many times as needed
        demo += [base_demo[-1] for i in range(max_steps_per_episode - len(base_demo) + 1)]
    elif extension_type == "extend_all":
        demo = []
        for state in base_demo:
            if demo_extend_num is None:
                # extend thru end of episode
                demo += [state for i in range(math.ceil(max_steps_per_episode / len(base_demo)))]
            else: 
                demo += [state for i in range(demo_extend_num)]

        if len(demo) < max_steps_per_episode:
            demo += [base_demo[-1] for i in range(max_steps_per_episode - len(base_demo) + 1)]

    assert len(demo) >= max_steps_per_episode
    if time_feat:
        time_feature = np.expand_dims(1 - (np.arange(len(demo)) / max_steps_per_episode), axis=1)
        demo = np.concatenate((np.array(demo), time_feature), axis=1) # NOTE THAT THIS CASTS THE DEMO TO np.ndarray TYPE

    return demo
