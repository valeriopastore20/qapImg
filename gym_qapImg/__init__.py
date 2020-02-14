from gym.envs.registration import register

register(
    id='qapImg-v0',
    entry_point='gym_qapImg.envs:QapImgEnv',
)