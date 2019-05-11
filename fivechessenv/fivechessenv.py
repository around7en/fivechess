import gym
import logging
import numpy
import random
from gym import spaces
 
 
logger = logging.getLogger(__name__)
 
class FiveChessEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 2
	}
 
	def __init__(self):
		#棋盘大小
		self.SIZE = 8
		#初始棋盘是0    -1表示黑棋子   1表示白棋子
		self.chessboard = [ [  0 for v in range(self.SIZE)  ] for v in range(self.SIZE) ]
		self.viewer = None
		self.step_count = 0
 
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
 
 
	def is_valid_coord(self,x,y):
		return x>=0 and x<self.SIZE and y>=0 and y<self.SIZE
 
	def is_valid_set_coord(self,x,y):
		return self.is_valid_coord(x,y) and self.chessboard[x][y]==0
 
	#返回一个有效的下棋位置
	def get_valid_pos_weights(self):
		results = []
		for x in range(self.SIZE):
			for y in range(self.SIZE):
				if self.chessboard[x][y]==0:
					results.append(1)
				else:
					results.append(0)
		return results
 
	#action 包括坐标和棋子颜色  例如：[1,3,1] 表示： 坐标（1,3），白棋
	#输出 下一个状态，动作价值，是否结束，额外信息{}
	def step(self, action):
		'''
		#非法操作
		if not self.is_valid_set_coord(action[0],action[1]):
			return self.chessboard,-50,False,{}
		'''
 
		#棋子
		self.chessboard[action[0]][action[1]] = action[2]
 
		self.step_count +=1
 
		#胜负判定
		color = action[2]
		
		win_reward = 1000
		common_reward = -20
		draw_reward = 0
 
		#1.横向
		count = 1
		win = False
 
		i = 1
		stop0 = False
		stop1 = False
 
		while i<self.SIZE:
			x = action[0]+i
			y = action[1]
			#左边
			if (not stop0) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop0 = True
			#右边
			x = action[0]-i
			if (not stop1) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop1 = True
 
			#超过5个相同，则胜利
			if count>=5:
				win = True
				break
 
			#都不相同，停止探索
			if stop0 and stop1:
				break
			i+=1
 
		if win:
			print('win1')
			return self.chessboard,win_reward,True,{}
		#2.纵向
		count = 1
		win = False
 
		i = 1
		stop0 = False
		stop1 = False
 
		while i<self.SIZE:
			x = action[0]
			y = action[1]+i
			#左边
			if (not stop0) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop0 = True
			#右边
			y = action[1]-i
			if (not stop1) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop1 = True
 
			#超过5个相同，则胜利
			if count>=5:
				win = True
				break
 
			#都不相同，停止探索
			if stop0 and stop1:
				break
			i+=1
		if win:
			print('win2')
			return self.chessboard,win_reward,True,{}
		#3.左斜向
		count = 1
		win = False
 
		i = 1
		stop0 = False
		stop1 = False
 
		while i<self.SIZE:
			x = action[0]+i
			y = action[1]+i
			#左边
			if (not stop0) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop0 = True
			#右边
			x = action[0]-i
			y = action[1]-i
			if (not stop1) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop1 = True
 
			#超过5个相同，则胜利
			if count>=5:
				win = True
				break
 
			#都不相同，停止探索
			if stop0 and stop1:
				break
			i+=1
		if win:
			print('win3')
			return self.chessboard,win_reward,True,{}
 
		#3.右斜向
		count = 1
		win = False
 
		i = 1
		stop0 = False
		stop1 = False
 
		while i<self.SIZE:
			x = action[0]-i
			y = action[1]+i
			#左边
			if (not stop0) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop0 = True
			#右边
			x = action[0]+i
			y = action[1]-i
			if (not stop1) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop1 = True
 
			#超过5个相同，则胜利
			if count>=5:
				win = True
				break
 
			#都不相同，停止探索
			if stop0 and stop1:
				break
			i+=1
		if win:
			print('win4')
			return self.chessboard,win_reward,True,{}
 
		if self.step_count == self.SIZE*self.SIZE:
			print('draw')
			return self.chessboard,draw_reward,True,{}
 
		return self.chessboard,common_reward,False,{}
 
	def reset(self):
		self.chessboard = [ [  0 for v in range(self.SIZE)  ] for v in range(self.SIZE) ]
		self.step_count = 0
		return self.chessboard
 
	def render(self, mode = 'human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return
 
		screen_width = 800
		screen_height = 800
		space = 10
		width = (screen_width - space*2)/(self.SIZE-1)
 
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			bg = rendering.FilledPolygon([(0,0),(screen_width,0),(screen_width,screen_height),(0,screen_height),(0,0)])
			bg.set_color(0.2,0.2,0.2)
			self.viewer.add_geom(bg)
			
			#棋盘网格
			for i in range(self.SIZE):
				line = rendering.Line((space,space+i*width),(screen_width-space,space+i*width))
				line.set_color(1, 1, 1)
				self.viewer.add_geom(line)
			for i in range(self.SIZE):
				line = rendering.Line((space+i*width,space),(space+i*width,screen_height - space))
				line.set_color(1, 1, 1)
				self.viewer.add_geom(line)
				
			#棋子
			self.chess = []
			for x in range(self.SIZE):
				self.chess.append([])
				for y in range(self.SIZE):
					c = rendering.make_circle(width/2-3)
					ct = rendering.Transform(translation=(-10,-10))
					c.add_attr(ct)
					c.set_color(0,0,0)
					self.chess[x].append([c,ct])
					self.viewer.add_geom(c)
 
			
 
		for x in range(self.SIZE):
			for y in range(self.SIZE):	
				if self.chessboard[x][y]!=0:
					self.chess[x][y][1].set_translation(space+x*width,space+y*width)
					if self.chessboard[x][y]==1:
						self.chess[x][y][0].set_color(255,255,255)
					else:
						self.chess[x][y][0].set_color(0,0,0)
				else:
					self.chess[x][y][1].set_translation(-50,-50)
 
 
		return self.viewer.render(return_rgb_array=mode == 'rgb_array')
 
		#if self.state is None: return None
		#return super().render(mode)
