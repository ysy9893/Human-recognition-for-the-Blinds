#클래스 구분
#https://pythonhosted.org/pyglet/api/pyglet.media.Listener-class.html
import pyglet
player = pyglet.media.Player()
test_sound = pyglet.media.StaticSource(pyglet.media.load('test.wav'))
player.queue(test_sound)
player.position=(100,0,0)
player.play()
pyglet.app.event_loop.sleep(10)
