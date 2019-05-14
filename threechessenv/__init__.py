from gym.envs.registration import register
register(
    id='ThreeChessEnv-v0',
    entry_point='threechessenv.threechessenv:ThreeChessEnv', #第一个env是文件夹名字，第二个env是文件名字，第三个是文件内类的名字
)