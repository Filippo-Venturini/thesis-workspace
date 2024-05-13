from magent2.gridworld import GridWorld
from magent2.gridworld import Config

configuration = Config()
args = dict(map_width = 10, map_height = 10)
configuration.set(args) #10, 10, 2, "/render/", 12345, False, False, False

gridWorld = GridWorld(configuration)

gridWorld.render()