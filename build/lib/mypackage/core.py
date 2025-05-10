from .version import __version__

def hello(to): 
    print('hello %s; (code version: %s)'%(to, __version__) )